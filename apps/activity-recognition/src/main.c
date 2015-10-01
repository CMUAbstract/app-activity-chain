#include <msp430.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <math.h>

#include <wisp-base.h>
#include <accel.h>
#include <spi.h>

#include <libchain/chain.h>

// Peripheral registers not defined by wisp-base (or defined inconveniently)
#define PAUXIN      P3IN
#define PAUXOUT     P3OUT
#define PAUXDIR     P3DIR
#define PAUXREN     P3REN

#define USE_LEDS
#define FLASH_ON_BOOT

#define MODEL_SIZE 95

// Number of samples to discard before recording training set
#define NUM_WARMUP_SAMPLES 10
#define TRANING_SET_SIZE MODEL_SIZE

#define ACCEL_WINDOW_SIZE 4
#define MODEL_COMPARISONS 10

// number of samples until experiment is "idle", the computed
// results (moving/stationary stats) are "output" to non-volatile
// memory, and the LEDs go on
#define SAMPLES_TO_COLLECT 10000

// two features: mean & stdev
#define NUM_FEATURES 2

#define SEC_TO_CYCLES 4000000 /* 4 MHz */

#define IDLE_BLINK_RATE SEC_TO_CYCLES

typedef threeAxis_t_8 accelReading;
typedef accelReading accelWindow[ACCEL_WINDOW_SIZE];

typedef struct {
    unsigned meanmag;
    unsigned stddevmag;
} features_t;

typedef enum {
    CLASS_STATIONARY,
    CLASS_MOVING,
} class_t;

typedef enum {
    MODE_IDLE = 0,
    MODE_TRAIN_STATIONARY = PIN_AUX1,
    MODE_TRAIN_MOVING = PIN_AUX2,
    MODE_ACQUIRE = (PIN_AUX1 | PIN_AUX2),
} run_mode_t;

// We support using a model that is either
//   (1) acquired at runtime (via training phase)
//   (2) hardcoded
//
// This is the hardcoded model data.

volatile __fram const unsigned hardcoded_model_data_stationary[] = {
#include "int_wisp5_stationary.h"
};
volatile __fram const unsigned hardcoded_model_data_moving[] = {
#include "int_wisp5_flapping.h"
};

features_t *hardcoded_model_stationary = (features_t *)hardcoded_model_data_stationary;
features_t *hardcoded_model_moving     = (features_t *)hardcoded_model_data_moving;

/* End-results of the program are output into non-volatile memory.
 * Think of this as the "display" where we show the result of the long
 * (intermittent) computation. */
volatile __fram float resultMovingPct;
volatile __fram float resultStationaryPct;

typedef struct {
    CHAN_FIELD(unsigned, totalCount);
    CHAN_FIELD(unsigned, movingCount);
    CHAN_FIELD(unsigned, stationaryCount);
} msg_stats;

typedef struct {
    CHAN_FIELD(unsigned, trainingSetSize);
} msg_train;

typedef struct {
    CHAN_FIELD(unsigned, samplesInWindow);
} msg_windowSize;

typedef struct {
    CHAN_FIELD(unsigned, discardedSamplesCount);
} msg_warmup;

typedef struct {
    CHAN_FIELD(accelReading, window[ACCEL_WINDOW_SIZE]);
} msg_window;

typedef struct {
    // TODO: think more about struct types in channels, for now separate arrays
    CHAN_FIELD(features_t, model_stationary[MODEL_SIZE]);
    CHAN_FIELD(features_t, model_moving[MODEL_SIZE]);
} msg_model;

typedef struct {
    CHAN_FIELD(run_mode_t, mode);
} msg_mode;

typedef struct {
    CHAN_FIELD(run_mode_t, class);
} msg_class;

typedef struct {
    CHAN_FIELD(features_t, features);
} msg_features;

TASK(0, task_init)
TASK(1, task_selectMode)
TASK(2, task_resetStats)
TASK(3, task_sample)
TASK(4, task_featurize)
TASK(5, task_classify)
TASK(6, task_stats)
TASK(7, task_warmup)
TASK(8, task_train)
TASK(9, task_idle)

CHANNEL(task_init, task_classify, msg_model);

CHANNEL(task_selectMode, task_warmup, msg_warmup);
CHANNEL(task_selectMode, task_featurize, msg_mode);
CHANNEL(task_selectMode, task_train, msg_class);

CHANNEL(task_resetStats, task_stats, msg_stats);
CHANNEL(task_resetStats, task_sample, msg_windowSize);

CHANNEL(task_sample, task_featurize, msg_window);
SELF_CHANNEL(task_sample, msg_windowSize);

CHANNEL(task_featurize, task_train, msg_features);
CHANNEL(task_featurize, task_classify, msg_features);

CHANNEL(task_classify, task_stats, msg_class);

SELF_CHANNEL(task_stats, msg_stats);

SELF_CHANNEL(task_warmup, msg_warmup);
CHANNEL(task_warmup, task_train, msg_train);

SELF_CHANNEL(task_train, msg_train);
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
    threeAxis_t_8 accelID;

    // Unlock I/O ports: required after boot up
    PM5CTL0 &= ~LOCKLPM5;

    setupDflt_IO();

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

    // AUX pins select run mode: configure as inputs with pull-ups
    PAUXDIR &= ~(PIN_AUX1 | PIN_AUX2);
    PAUXREN |= PIN_AUX1 | PIN_AUX2;
    PAUXOUT &= ~(PIN_AUX1 | PIN_AUX2); // pull-down

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
    unsigned i;

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
    for (i = 0; i < MODEL_SIZE; ++i) {
        CHAN_OUT(model_stationary[i], hardcoded_model_stationary[i], CH(task_init, task_classify));
        CHAN_OUT(model_moving[i], hardcoded_model_moving[i], CH(task_init, task_classify));
    }

    TRANSITION_TO(task_selectMode);
}
void task_selectMode()
{
    uint8_t pin_state = (PAUXIN & (PIN_AUX1 | PIN_AUX2));
    switch(pin_state) {
        case MODE_TRAIN_STATIONARY:
            CHAN_OUT(discardedSamplesCount, 0, CH(task_selectMode, task_warmup));
            CHAN_OUT(mode, MODE_TRAIN_STATIONARY, CH(task_selectMode, task_featurize));
            CHAN_OUT(class, CLASS_STATIONARY, CH(task_selectMode, task_train));

            TRANSITION_TO(task_warmup);
            break;

        case MODE_TRAIN_MOVING:
            CHAN_OUT(discardedSamplesCount, 0, CH(task_selectMode, task_warmup));
            CHAN_OUT(mode, MODE_TRAIN_MOVING, CH(task_selectMode, task_featurize));
            CHAN_OUT(class, CLASS_MOVING, CH(task_selectMode, task_train));

            TRANSITION_TO(task_warmup);
            break;

        case MODE_ACQUIRE:
            CHAN_OUT(mode, MODE_ACQUIRE, CH(task_selectMode, task_featurize));

            TRANSITION_TO(task_resetStats);
            break;

        default:
            TRANSITION_TO(task_idle);
    }
}

void task_resetStats()
{
    // NOTE: could roll this into selectMode task, but no compelling reason

    // NOTE: not combined into one struct because not all code paths use both
    CHAN_OUT(movingCount, 0, CH(task_resetStats, task_stats));
    CHAN_OUT(stationaryCount, 0, CH(task_resetStats, task_stats));

    TRANSITION_TO(task_sample);
}

void task_sample()
{
    accelReading sample;
    unsigned samplesInWindow;

    ACCEL_singleSample(&sample);

    samplesInWindow = *CHAN_IN(samplesInWindow,
                               CH(task_resetStats, task_sample),
                               SELF_IN_CH(task_sample));

    samplesInWindow++;
    CHAN_OUT(window[samplesInWindow], sample, CH(task_sample, task_featurize));

    if (samplesInWindow < ACCEL_WINDOW_SIZE) {
        CHAN_OUT(samplesInWindow, samplesInWindow, SELF_OUT_CH(task_sample));
        TRANSITION_TO(task_sample);
    } else {
        CHAN_OUT(samplesInWindow, 0, SELF_OUT_CH(task_sample));
        TRANSITION_TO(task_featurize);
    }
}

void task_featurize()
{
   accelReading *reading;
   accelReading mean, stddev;
   features_t features;
   run_mode_t mode;
 
   mean.x = mean.y = mean.z = 0;
   stddev.x = stddev.y = stddev.z = 0;
   int i;
   for (i = 0; i < ACCEL_WINDOW_SIZE; i++) {
       reading = CHAN_IN1(window[i], CH(task_sample, task_featurize));
       mean.x += reading->x;
       mean.y += reading->y;
       mean.z += reading->z;
   }
   /*
   mean[0] = mean[0] / ACCEL_WINDOW_SIZE;
   mean[1] = mean[1] / ACCEL_WINDOW_SIZE;
   mean[2] = mean[2] / ACCEL_WINDOW_SIZE;
   */
   mean.x >>= 2;
   mean.y >>= 2;
   mean.z >>= 2;
 
   for (i = 0; i < ACCEL_WINDOW_SIZE; i++) {
       // TODO: room for optimization: promotion to volatile (since same vals read above)
       reading = CHAN_IN1(window[i], CH(task_sample, task_featurize));
       stddev.x += reading->x > mean.x ? reading->x - mean.x
                                      : mean.x - reading->x;
       stddev.y += reading->y > mean.y ? reading->y - mean.y
                                    : mean.y - reading->y;
       stddev.z += reading->z > mean.z ? reading->z - mean.z
                                      : mean.z - reading->z;
   }
   /*
   stddev[0] = stddev[0] / (ACCEL_WINDOW_SIZE - 1);
   stddevy = stddevy / (ACCEL_WINDOW_SIZE - 1);
   stddev.z = stddev.z / (ACCEL_WINDOW_SIZE - 1);
   */
   stddev.x >>= 2;
   stddev.y >>= 2;
   stddev.z >>= 2;
 
   float meanmag_f = (float)
     ((mean.x*mean.x) + (mean.y*mean.y) + (mean.z*mean.z));
   float stddevmag_f = (float)
     ((stddev.x*stddev.x) + (stddev.y*stddev.y) + (stddev.z*stddev.z));
 
   meanmag_f   = sqrtf(meanmag_f);
   stddevmag_f = sqrtf(stddevmag_f);
 
   features.meanmag   = (long)meanmag_f;
   features.stddevmag = (long)stddevmag_f;
 
   mode = *CHAN_IN1(mode, CH(task_selectMode, task_featurize));

   switch (mode) {
       case MODE_TRAIN_STATIONARY:
       case MODE_TRAIN_MOVING:
           CHAN_OUT(features, features, CH(task_featurize, task_train));
           TRANSITION_TO(task_train);
           break;
        case MODE_ACQUIRE:
           CHAN_OUT(features, features, CH(task_featurize, task_classify));
           TRANSITION_TO(task_classify);
           break;
        default:
           // TODO: abort
           break;
    }
 }
 
void task_classify() {
    int move_less_error = 0;
    int stat_less_error = 0;
    int i;
    class_t class;
    features_t features;
    long int meanmag;
    long int stddevmag;
    features_t model_features;
  
    features = *CHAN_IN1(features, CH(task_featurize, task_classify));

    // TODO: does it make sense to get a reference to the whole model at once?

    // TODO: use features obj directly
    meanmag = features.meanmag;
    stddevmag = features.stddevmag;
  
    for (i = 0; i < MODEL_COMPARISONS; i += NUM_FEATURES) {
        model_features = *CHAN_IN(model_stationary[i], CH(task_init, task_classify),
                                                    CH(task_train, task_classify));
        long int stat_mean_err = (model_features.meanmag > meanmag)
            ? (model_features.meanmag - meanmag)
            : (meanmag - model_features.meanmag);

        long int stat_sd_err = (model_features.stddevmag > stddevmag)
            ? (model_features.stddevmag - stddevmag)
            : (stddevmag - model_features.stddevmag);

        model_features = *CHAN_IN(model_moving[i], CH(task_init, task_classify),
                                                CH(task_train, task_classify));
        long int move_mean_err = (model_features.meanmag > meanmag)
            ? (model_features.meanmag - meanmag)
            : (meanmag - model_features.meanmag);

        long int move_sd_err = (model_features.stddevmag > stddevmag)
            ? (model_features.stddevmag - stddevmag)
            : (stddevmag - model_features.stddevmag);

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
  
    class = (move_less_error > stat_less_error) ? CLASS_MOVING : CLASS_STATIONARY;
    CHAN_OUT(class, class, CH(task_classify, task_stats));
  
    TRANSITION_TO(task_stats);
}

void task_stats()
{
    unsigned totalCount = 0, movingCount = 0, stationaryCount = 0;
    class_t class;

    totalCount = *CHAN_IN(totalCount, CH(task_resetStats, task_stats),
                                      SELF_IN_CH(task_stats));
    
    totalCount++;

    CHAN_OUT(totalCount, totalCount, SELF_OUT_CH(task_stats));

    class = *CHAN_IN1(class, CH(task_classify, task_stats));

    if (class) {

#if defined (USE_LEDS)
      PJOUT &= ~BIT6;
      P4OUT |= BIT0;
#endif //USE_LEDS

      movingCount = *CHAN_IN(movingCount, CH(task_resetStats, task_stats),
                                          SELF_IN_CH(task_stats));
      movingCount++;
      CHAN_OUT(movingCount, movingCount, SELF_OUT_CH(task_stats));
    } else {

#if defined (USE_LEDS)
      P4OUT &= ~BIT0;  // Toggle P1.0 using exclusive-OR
      PJOUT |= BIT6;
#endif //USE_LEDS

      stationaryCount = *CHAN_IN(stationaryCount, CH(task_resetStats, task_stats),
                                                  SELF_IN_CH(task_stats));
      stationaryCount++;
      CHAN_OUT(stationaryCount, stationaryCount, SELF_OUT_CH(task_stats));
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

        TRANSITION_TO(task_idle);
    } else {
        TRANSITION_TO(task_sample);
    }
}

void task_warmup()
{
    unsigned discardedSamplesCount;
    threeAxis_t_8 sample;

    discardedSamplesCount = *CHAN_IN(discardedSamplesCount,
                                     CH(task_selectMode, task_warmup),
                                     SELF_IN_CH(task_warmup));

    if (discardedSamplesCount < NUM_WARMUP_SAMPLES) {

        // Re-using the sample task is possible, but it might not be desirable:
        // half the work in sample task is filling the window, which is
        // not relevant in training mode. Also, if re-used, the sample task
        // will need to choose which task to go to next based on training/normal
        // mode, which would need to be channeled to it.
        ACCEL_singleSample(&sample);
    
        discardedSamplesCount++;
        CHAN_OUT(discardedSamplesCount, discardedSamplesCount, SELF_OUT_CH(task_warmup));
        TRANSITION_TO(task_warmup);
    } else {
        CHAN_OUT(trainingSetSize, 0, CH(task_warmup, task_train));
        TRANSITION_TO(task_sample);
    }
}

void task_train()
{
    features_t features;
    unsigned trainingSetSize;;
    unsigned class;

    features = *CHAN_IN1(features, CH(task_featurize, task_train));
    trainingSetSize = *CHAN_IN(trainingSetSize, CH(task_warmup, task_train),
                                                SELF_IN_CH(task_train));
    class = *CHAN_IN1(class, CH(task_selectMode, task_train));

    if (trainingSetSize < TRANING_SET_SIZE) {
        switch (class) {
            case CLASS_STATIONARY: 
                CHAN_OUT(model_stationary[trainingSetSize], features, CH(task_train, task_classify));
                break;
            case CLASS_MOVING: 
                CHAN_OUT(model_moving[trainingSetSize], features, CH(task_train, task_classify));
                break;
        }

        trainingSetSize++;
        CHAN_OUT(trainingSetSize, trainingSetSize, SELF_IN_CH(task_train));
        TRANSITION_TO(task_sample);
    } else {
        TRANSITION_TO(task_idle);
    }
}

void task_idle() {
    unsigned i;

    P4OUT ^= BIT0;
    PJOUT ^= BIT6;

    for (i = 0; i < IDLE_BLINK_RATE / 2 / (1U << 15); ++i)
        __delay_cycles(1U << 15);

    TRANSITION_TO(task_selectMode);
}

INIT_FUNC(initializeHardware)
ENTRY_TASK(task_init)
