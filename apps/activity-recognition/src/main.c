#include <msp430.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <wisp5.h>
#include <accel.h>
#include <spi.h>
#include <math.h>

#include <libchain/chain.h>

#define USE_LEDS
#define FLASH_ON_BOOT

#define MODEL_SIZE 190

// Number of samples to discard before recording training set
#define NUM_WARMUP_SAMPLES 10
#define TRANING_SET_SIZE MODEL_SIZE

#define ACCEL_WINDOW_SIZE 4
#define MODEL_COMPARISONS 10

// number of samples until experiment is "done", the computed
// results (moving/stationary stats) are "output" to non-volatile
// memory, and the LEDs go on
#define SAMPLES_TO_COLLECT 10000

// two features: mean & stdev
#define NUM_FEATURES 2

#define SEC_TO_CYCLES 4000000 /* 4 MHz */

typedef long int accelReading[3];
typedef accelReading accelWindow[ACCEL_WINDOW_SIZE];

typedef struct {
    unsigned meanmag;
    unsigned stddevmag;
} features_t;

typedef struct {
    unsigned stationary[MODEL_SIZE];
    unsigned moving[MODEL_SIZE];
} model_t;

typedef enum {
    MODE_TRAIN,
    MODE_TRAIN_STATIONARY,
    MODE_TRAIN_MOVING,
    MODE_ACQUIRE,
} mode_t;

// We support using a model that is either
//   (1) acquired at runtime (via training phase)
//   (2) hardcoded
//
// This is the hardcoded model data.
static const model_t hardcoded_model = {
    .stationary = {
#include "int_wisp5_stationary.h"
    },
    .moving = {
#include "int_wisp5_flapping.h"
    }
}:

/* End-results of the program are output into non-volatile memory.
 * Think of this as the "display" where we show the result of the long
 * (intermittent) computation. */
volatile __fram float resultMovingPct;
volatile __fram float resultStationaryPct;

typedef struct {
    CHAN_FIELD(unsigned, movingCount);
    CHAN_FIELD(unsigned, stationaryCount);
} msg_stats;

typedef struct {
    CHAN_FIELD(unsigned, totalCount);
} msg_count;

typedef struct {
    CHAN_FIELD(unsigned, trainingSetSize);
} msg_train;

typedef struct {
    CHAN_FIELD(unsigned, samplesInWindow);
} msg_windowSize;

typedef struct {
    CHAN_FIELD(unsigned, discardedSamplesSize);
} msg_warmup;

typedef struct {
    CHAN_FIELD(accelWindow, window);
} msg_window;

typedef struct {
    CHAN_FIELD(model_t, model);
} msg_model;

typedef struct {
    CHAN_FIELD(mode_t, mode);
} msg_mode;

typedef struct {
    CHAN_FIELD(mode_t, class);
} msg_class;

typedef struct {
    CHAN_FIELD(features_t, features);
} msg_features;

CHANNEL(task_init, task_classify, msg_model);

CHANNEL(task_selectMode, task_warmup, msg_warmup);
CHANNEL(task_selectMode, task_featurize, msg_mode);
CHANNEL(task_selectMode, task_train, msg_class);

CHANNEL(task_resetStats, task_stats, msg_stats);

CHANNEL(task_sample, task_featurize, msg_window);
CHANNEL(task_sample, task_sample, msg_windowSize);

CHANNEL(task_featurize, task_train, msg_features);
CHANNEL(task_featurize, task_classify, msg_features);

CHANNEL(task_classify, task_stats, msg_stats);

CHANNEL(task_stats, task_stats, msg_stats);

CHANNEL(task_warmup, task_warmup, msg_warmup);
CHANNEL(task_warmup, task_train, msg_train);

CHANNEL(task_train, task_train, msg_train);
CHANNEL(task_train, task_classify, msg_model);

#ifdef __clang__
void __delay_cycles(unsigned long cyc) {
  unsigned i;
  for (i = 0; i < (cyc >> 3); ++i)
    ;
}
#endif

void initializeHardware()
{
    threeAxis_t accelID;

    setupDflt_IO();

    PRXEOUT |=
        PIN_RX_EN; /** TODO: enable PIN_RX_EN only when needed in the future */

    // set clock speed to 4 MHz
    CSCTL0_H = 0xA5;
    CSCTL1 = DCOFSEL0 | DCOFSEL1;
    CSCTL2 = SELA_0 | SELS_3 | SELM_3;
    CSCTL3 = DIVA_0 | DIVS_0 | DIVM_0;

    /*Before anything else, do the device hardware configuration*/
    P4DIR |= BIT0;
    PJDIR |= BIT6;

#if defined(USE_LEDS) && defined(FLASH_ON_BOOT)
    P4OUT |= BIT0;
    PJOUT |= BIT6;
    __delay_cycles(0xffff);
    P4OUT &= ~BIT0;
    PJOUT &= ~BIT6;
#endif

    /*
    SPI_initialize();
    ACCEL_initialize();
    */
    // ACCEL_SetReg(0x2D,0x02);

    /* TODO: move the below stuff to accel.c */
    BITSET(PDIR_AUX3, PIN_AUX3);
    __delay_cycles(50);
    BITCLR(P1OUT, PIN_AUX3);
    __delay_cycles(50);
    BITSET(P1OUT, PIN_AUX3);
    __delay_cycles(50);

    BITSET(P4SEL1, PIN_ACCEL_EN);
    BITSET(P4SEL0, PIN_ACCEL_EN);

    BITSET(P2SEL1, PIN_ACCEL_SCLK | PIN_ACCEL_MISO | PIN_ACCEL_MOSI);
    BITCLR(P2SEL0, PIN_ACCEL_SCLK | PIN_ACCEL_MISO | PIN_ACCEL_MOSI);
    __delay_cycles(5);
    SPI_initialize();
    __delay_cycles(5);
    ACCEL_range();
    __delay_cycles(5);
    ACCEL_initialize();
    __delay_cycles(5);
    ACCEL_readID(&accelID);
}

void task_init()
{
    // Until training happens, we use the hardcoded model. To keep
    // the classify task unaware of where the model came from, we
    // we channel the respective model to it. To be able to channel
    // the hardcoded model to classify later, some task must
    // first write it into a channel. This is a copy and is controvertial.
    //
    // It might be possible to arrange for this "copy" to happen at compile
    // time: it is essentially an initialization of a variable. Perhaps,
    // a channel declaration could take an initial value.
    //
    // As it stands now this is a struct assignment: a copy.
    CHAN_OUT(task_init, task_classify, model, hardcoded_model);

    transition_to(task_selectMode);
}

void task_selectMode()
{

    switch(pin_state) {
        case MODE_TRAIN_STATIONARY:
            CHAN_OUT(task_selectMode, task_warmup, discardedSamplesCount, 0);
            CHAN_OUT(task_selectMode, task_featurize, mode, MODE_TRAIN);
            CHAN_OUT(task_selectMode, task_train, class, CLASS_STATIONARY);

            transition_to(task_warmup);
            break;

        case MODE_TRAIN_MOVING:
            CHAN_OUT(task_selectMode, task_warmup, discardedSamplesCount, 0);
            CHAN_OUT(task_selectMode, task_featurize, mode, MODE_TRAIN);
            CHAN_OUT(task_selectMode, task_train, class, CLASS_MOVING);

            transition_to(task_warmup);
            break;

        case MODE_ACQUIRE:
            CHAN_OUT(task_selectMode, task_featurize, mode, MODE_ACQUIRE);

            transition_to(task_resetStats);
            break;
        default:
            transition_to(task_done);
    }
}

void task_resetStats()
{
    // NOTE: could roll this into selectMode task, but no compelling reason

    // NOTE: not combined into one struct because not all code paths use both
    CHAN_OUT(task_resetStats, task_stats, movingCount, 0);
    CHAN_OUT(task_resetStats, task_stats, stationaryCount, 0);

    transition_to(task_sample);
}

void task_sample()
{
    threeAxis_t sample;

    ACCEL_singleSample(&sample);

    samplesInWindow = *CHAN_IN2(samplesInWindow, task_sample, task_resetStats, task_sample);

    samplesInWindow++;
    CHAN_OUT(task_sample, task_featurize, window[samplesInWindow], sample);

    if (samplesInWindow < ACCEL_WINDOW_SIZE) {
        CHAN_OUT(task_sample, task_sample, samplesInWindow, samplesInWindow);
        transition_to(task_sample);
    } else {
        CHAN_OUT(task_sample, task_sample, samplesInWindow, 0);
        transition_to(task_featurize);
    }
}

void task_featurize()
{
   accelWindow aWin;
   long int meanmag;
   long int stddevmag;
   features_t features;
   
   aWin = CHAN_IN1(window, task_featurize, task_sample);
 
   mean[0] = mean[1] = mean[2] = 0;
   stddev[0] = stddev[1] = stddev[2] = 0;
   int i;
   for (i = 0; i < ACCEL_WINDOW_SIZE; i++) {
     mean[0] += aWin[i][0];  // x
     mean[1] += aWin[i][1];  // y
     mean[2] += aWin[i][2];  // z
   }
   /*
   mean[0] = mean[0] / ACCEL_WINDOW_SIZE;
   mean[1] = mean[1] / ACCEL_WINDOW_SIZE;
   mean[2] = mean[2] / ACCEL_WINDOW_SIZE;
   */
   mean[0] >>= 2;
   mean[1] >>= 2;
   mean[2] >>= 2;
 
   for (i = 0; i < ACCEL_WINDOW_SIZE; i++) {
     stddev[0] += aWin[i][0] > mean[0] ? aWin[i][0] - mean[0]
                                       : mean[0] - aWin[i][0];  // x
     stddev[1] += aWin[i][1] > mean[1] ? aWin[i][1] - mean[1]
                                       : mean[1] - aWin[i][1];  // y
     stddev[2] += aWin[i][2] > mean[2] ? aWin[i][2] - mean[2]
                                       : mean[2] - aWin[i][2];  // z
   }
   /*
   stddev[0] = stddev[0] / (ACCEL_WINDOW_SIZE - 1);
   stddev[1] = stddev[1] / (ACCEL_WINDOW_SIZE - 1);
   stddev[2] = stddev[2] / (ACCEL_WINDOW_SIZE - 1);
   */
   stddev[0] >>= 2;
   stddev[1] >>= 2;
   stddev[2] >>= 2;
 
   float meanmag_f = (float)
     ((mean[0]*mean[0]) + (mean[1]*mean[1]) + (mean[2]*mean[2]));
   float stddevmag_f = (float)
     ((stddev[0]*stddev[0]) + (stddev[1]*stddev[1]) + (stddev[2]*stddev[2]));
 
   meanmag_f   = sqrtf(meanmag_f);
   stddevmag_f = sqrtf(stddevmag_f);
 
   features.meanmag   = (long)meanmag_f;
   features.stddevmag = (long)stddevmag_f;
 
   mode = *CHAN_IN1(mode, task_featurize, task_selectMode);

   switch (mode) {
       case MODE_TRAIN:
           CHAN_OUT(task_featurize, task_train, features, features);
           transition_to(task_train);
           break;
        case MODE_ACQUIRE:
           CHAN_OUT(task_featurize, task_classify, features, features);
           transition_to(task_classify);
           break;
    }
 }
 
int task_classify() {
    int move_less_error = 0;
    int stat_less_error = 0;
    int i;
    bool class;
    features_t features;
    long int meanmag;
    long int stddevmag;
    model_t *model;
  
    features = *CHAN_IN1(features, task_classify, task_featurize);
    model = CHAN_IN2(model, task_classify, task_train, task_init);

    // TODO: use features obj directly
    meanmag = features.meanmag;
    stddevmag = features.stddevmag;
  
    for (i = 0; i < MODEL_COMPARISONS; i += NUM_FEATURES) {
      long int stat_mean_err = (model->stationary[i] > meanmag)
                                   ? (model->stationary[i] - meanmag)
                                   : (meanmag - model->stationary[i]);
  
      long int stat_sd_err = (model->stationary[i + 1] > stddevmag)
                                 ? (model->stationary[i + 1] - stddevmag)
                                 : (stddevmag - model->stationary[i + 1]);
  
      long int move_mean_err = (model->moving[i] > meanmag) ? (model->moving[i] - meanmag)
                                                      : (meanmag - model->moving[i]);
  
      long int move_sd_err = (model->moving[i + 1] > stddevmag)
                                 ? (model->moving[i + 1] - stddevmag)
                                 : (stddevmag - model->moving[i + 1]);
  
      if (move_mean_err < stat_mean_err) {
        move_less_error++;
      } else {
        stat_less_error++;
      }
  
      if (move_sd_err < stat_sd_err) {
        move_less_error++;
      } else {
        stat_less_error++;
      }
    }
  
    class = (move_less_error > stat_less_error);
    CHAN_OUT(task_classify, task_stats, class, class);
  
    transition_to(task_stats);
}

void task_stats()
{
    unsigned totalCount;
    unsigned class;

    totalCount = *CHAN_IN2(totalCount, task_stats, task_resetStats, task_stats);
    
    totalCount++;

    CHAN_OUT(task_stats, task_stats, totalCount, totalCount);

    class = *CHAN_IN1(class, task_stats, task_classify);

    if (class) {

#if defined (USE_LEDS)
      PJOUT &= ~BIT6;
      P4OUT |= BIT0;
#endif //USE_LEDS

      movingCount = *CHAN_IN1(movingCount, task_resetStats, task_stats);
      movingCount++;
      CHAN_OUT(task_stats, task_stats, movingCount);
    } else {

#if defined (USE_LEDS)
      P4OUT &= ~BIT0;  // Toggle P1.0 using exclusive-OR
      PJOUT |= BIT6;
#endif //USE_LEDS

      stationaryCount = *CHAN_IN1(stationaryCount, task_resetStats, task_stats);
      stationaryCount++;
      CHAN_OUT(task_stats, task_stats, stationaryCount);
    }

    if (totalCount > SAMPLES_TO_COLLECT) {
        // This is "I/O" (specifically, output). Yes, it's to non-volatile
        // memory, which may seem confusing, but for this program to have any
        // purpose, there needs to be I/O of the computed results to somewhere.
        // In channels model, this simply becomes explicit, because channel
        // non-volatile memory space has nothing to do with the memory space
        // where the results happen to be output.
        resultStationaryPct = ((float)stationaryCount / (float)totalCount) * 100.0f;
        resultMovingPct = ((float)movingCount / (float)totalCount) * 100.0f;

#if defined (USE_LEDS)
        P4OUT &= ~BIT0;
        PJOUT &= ~BIT6;
#endif

        transition_to(task_done);
    } else {
        transition_to(task_sample);
    }
}

void task_warmup()
{
    unsigned discardedSamplesCount;
    accelReading sample;

    discardedSamplesCount = *CHAN_IN2(discardedSamplesCount, task_warmup, task_selectMode, task_warmup);

    if (discardedSamplesCount < NUM_WARMUP_SAMPLES)

        // Re-using the sample task is possible, but it might not be desirable:
        // half the work in sample task is filling the window, which is
        // not relevant in training mode. Also, if re-used, the sample task
        // will need to choose which task to go to next based on training/normal
        // mode, which would need to be channeled to it.
        ACCEL_singleSample(&sample);
    
        discardedSamplesCount++;
        CHAN_OUT(task_warmup, task_warmup, discardedSamplesCount, discardedSamplesCount);
        transition_to(task_warmup);
    } else {
        CHAN_OUT(task_warmup, task_train, trainingSetSize, 0);
        transition_to(task_sample);
    }
}

void task_train()
{
    features_t features;
    unsigned trainingSetSize;;
    unsigned class;

    features = *CHAN_IN1(features, task_train, task_featurize);
    trainingSetSize = *CHAN_IN2(trainingSetSize, task_train, task_warmup, task_train);
    class = *CHAN_IN1(class, task_train, task_selectMode);

    if (trainingSetSize < TRANING_SET_SIZE) {
        switch (class) {
            case CLASS_STATIONARY: 
                CHAN_OUT(task_train, task_classify, model.stationary[trainingSetSize], features);
                break;
            case CLASS_MOVING: 
                CHAN_OUT(task_train, task_classify, model.moving[trainingSetSize], features);
                break;
        }

        trainingSetSize++;
        CHAN_OUT(task_train, task_train, trainingSetSize, trainingSetSize);
        transition_to(task_sample);
    } else {
        transition_to(task_done);
    }
}

void task_done() {
    unsigned i;

    P4OUT ^= BIT0;
    PJOUT ^= BIT6;

    for (i = 0; i < SEC_TO_CYCLES / (1U >> 15); ++i)
        __delay_cycles(1U >> 15);

    transition_to(task_selectMode);
}

INIT_FUNC(initializeHardware)
ENTRY_TASK(task_init)
