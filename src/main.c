#include <msp430.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <math.h>

#include <wisp-base.h>
#include <accel.h>
#include <spi.h>

#include <libchain/chain.h>

#include "pins.h"

#define SHOW_RESULT_ON_LEDS
#define SHOW_PROGRESS_ON_LEDS
#define SHOW_BOOT_ON_LEDS

#define ENABLE_PRINTF

#define MODEL_SIZE 95

// Number of samples to discard before recording training set
#define NUM_WARMUP_SAMPLES 5
#define TRANING_SET_SIZE MODEL_COMPARISONS

#define ACCEL_WINDOW_SIZE 4
#define MODEL_COMPARISONS 5
#define SAMPLE_NOISE_FLOOR 10 // TODO: made up value

// number of samples until experiment is "idle", the computed
// results (moving/stationary stats) are "output" to non-volatile
// memory, and the LEDs go on
#define SAMPLES_TO_COLLECT 4

// two features: mean & stdev
#define NUM_FEATURES 2

#define SEC_TO_CYCLES 4000000 /* 4 MHz */

#define IDLE_WAIT SEC_TO_CYCLES

#define IDLE_BLINKS 1
#define IDLE_BLINK_DURATION SEC_TO_CYCLES
#define SELECT_MODE_BLINKS  4
#define SELECT_MODE_BLINK_DURATION  (SEC_TO_CYCLES / 5)
#define SAMPLE_BLINKS  1
#define SAMPLE_BLINK_DURATION  (SEC_TO_CYCLES * 2)
#define FEATURIZE_BLINKS  2
#define FEATURIZE_BLINK_DURATION  (SEC_TO_CYCLES * 2)
#define CLASSIFY_BLINKS 1
#define CLASSIFY_BLINK_DURATION (SEC_TO_CYCLES * 4)
#define WARMUP_BLINKS 2
#define WARMUP_BLINK_DURATION (SEC_TO_CYCLES / 2)
#define TRAIN_BLINKS 1
#define TRAIN_BLINK_DURATION (SEC_TO_CYCLES * 4)

#define LED1 (1 << 0)
#define LED2 (1 << 1)

#ifndef ENABLE_PRINTF
#define printf(...)
#endif

// If you link-in wisp-base, then you have to define some symbols.
uint8_t usrBank[USRBANK_SIZE];

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
    MODE_TRAIN_STATIONARY = BIT(PIN_AUX_1),
    MODE_TRAIN_MOVING = BIT(PIN_AUX_2),
    MODE_ACQUIRE = (BIT(PIN_AUX_1) | BIT(PIN_AUX_2)),
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

struct msg_stats {
    CHAN_FIELD(unsigned, totalCount);
    CHAN_FIELD(unsigned, movingCount);
    CHAN_FIELD(unsigned, stationaryCount);
};

struct msg_train {
    CHAN_FIELD(unsigned, trainingSetSize);
};

struct msg_windowSize {
    CHAN_FIELD(unsigned, samplesInWindow);
};

struct msg_warmup {
    CHAN_FIELD(unsigned, discardedSamplesCount);
};

struct msg_window {
    CHAN_FIELD_ARRAY(accelReading, window, ACCEL_WINDOW_SIZE);
};

struct msg_model {
    // TODO: think more about struct types in channels, for now separate arrays
    CHAN_FIELD_ARRAY(features_t, model_stationary, MODEL_SIZE);
    CHAN_FIELD_ARRAY(features_t, model_moving, MODEL_SIZE);
};

struct msg_mode {
    CHAN_FIELD(run_mode_t, mode);
};

struct msg_class {
    CHAN_FIELD(run_mode_t, class);
};

struct msg_features {
    CHAN_FIELD(features_t, features);
};

TASK(0, task_init)
TASK(1, task_selectMode)
TASK(2, task_resetStats)
TASK(3, task_sample)
TASK(4, task_transform)
TASK(5, task_featurize)
TASK(6, task_classify)
TASK(7, task_stats)
TASK(8, task_warmup)
TASK(9, task_train)
TASK(10, task_idle)

CHANNEL(task_init, task_classify, msg_model);

CHANNEL(task_selectMode, task_warmup, msg_warmup);
CHANNEL(task_selectMode, task_featurize, msg_mode);
CHANNEL(task_selectMode, task_train, msg_class);
CHANNEL(task_selectMode, task_sample, msg_windowSize);

CHANNEL(task_resetStats, task_stats, msg_stats);
CHANNEL(task_resetStats, task_sample, msg_windowSize);

MULTICAST_CHANNEL(msg_window, ch_sample_window,
                 task_sample, task_transform, task_featurize);
SELF_CHANNEL(task_sample, msg_windowSize);
CHANNEL(task_transform, task_featurize, msg_window);

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

static void delay(uint32_t cycles)
{
    unsigned i;
    for (i = 0; i < cycles / (1U << 15); ++i)
        __delay_cycles(1U << 15);
}

static void blink(unsigned count, uint32_t duration, unsigned leds)
{
    unsigned i;
    for (i = 0; i < count; ++i) {
        GPIO(PORT_LED_1, OUT) |= (leds & LED1) ? BIT(PIN_LED_1) : 0x0;
        GPIO(PORT_LED_2, OUT) |= (leds & LED2) ? BIT(PIN_LED_2) : 0x0;
        delay(duration / 2);
        GPIO(PORT_LED_1, OUT) &= (leds & LED1) ? ~BIT(PIN_LED_1) : ~0x0;
        GPIO(PORT_LED_2, OUT) &= (leds & LED2) ? ~BIT(PIN_LED_2) : ~0x0;
        delay(duration / 2);
    }
}

void initializeHardware()
{
    threeAxis_t_8 accelID = {0};

    WISP_init();

    GPIO(PORT_LED_1, DIR) |= BIT(PIN_LED_1);
    GPIO(PORT_LED_2, DIR) |= BIT(PIN_LED_2);
#if defined(PORT_LED_3) // inidicates power-on when available
    GPIO(PORT_LED_3, DIR) |= BIT(PIN_LED_3);
    GPIO(PORT_LED_3, OUT) |= BIT(PIN_LED_3);
#endif


    UART_init();

#ifdef SHOW_BOOT_ON_LEDS
    GPIO(PORT_LED_1, OUT) |= BIT(PIN_LED_1);
    GPIO(PORT_LED_2, OUT) |= BIT(PIN_LED_2);
    delay(SEC_TO_CYCLES * 5);
    GPIO(PORT_LED_1, OUT) &= ~BIT(PIN_LED_1);
    GPIO(PORT_LED_2, OUT) &= ~BIT(PIN_LED_2);
#endif

    __enable_interrupt();

    printf("init: initializing accel\r\n");

    // AUX pins select run mode: configure as inputs with pull-ups
    GPIO(PORT_AUX, DIR) &= ~(BIT(PIN_AUX_1) | BIT(PIN_AUX_2));
    GPIO(PORT_AUX, OUT) &= ~(BIT(PIN_AUX_1) | BIT(PIN_AUX_2)); // pull-down
    GPIO(PORT_AUX, REN) |= BIT(PIN_AUX_1) | BIT(PIN_AUX_2);

    /*
    SPI_initialize();
    ACCEL_initialize();
    */
    // ACCEL_SetReg(0x2D,0x02);

    /* TODO: move the below stuff to accel.c */
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

    printf("init: accel hw id: 0x%x\r\n", accelID.x);
}

void task_init()
{
    unsigned i;

    printf("init\r\n");

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
#ifdef SHOW_PROGRESS_ON_LEDS
    blink(SELECT_MODE_BLINKS, SELECT_MODE_BLINK_DURATION, LED1 | LED2);
#endif

    uint8_t pin_state = GPIO(PORT_AUX, IN) & (BIT(PIN_AUX_1) | BIT(PIN_AUX_2));

    printf("selectMode: 0x%x\r\n", pin_state);

    switch(pin_state) {
        case MODE_TRAIN_STATIONARY:
            CHAN_OUT(discardedSamplesCount, 0, CH(task_selectMode, task_warmup));
            CHAN_OUT(mode, MODE_TRAIN_STATIONARY, CH(task_selectMode, task_featurize));
            CHAN_OUT(class, CLASS_STATIONARY, CH(task_selectMode, task_train));
            CHAN_OUT(samplesInWindow, 0, CH(task_selectMode, task_sample));

            TRANSITION_TO(task_warmup);
            break;

        case MODE_TRAIN_MOVING:
            CHAN_OUT(discardedSamplesCount, 0, CH(task_selectMode, task_warmup));
            CHAN_OUT(mode, MODE_TRAIN_MOVING, CH(task_selectMode, task_featurize));
            CHAN_OUT(class, CLASS_MOVING, CH(task_selectMode, task_train));
            CHAN_OUT(samplesInWindow, 0, CH(task_selectMode, task_sample));

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

    printf("resetStats\r\n");

    // NOTE: not combined into one struct because not all code paths use both
    CHAN_OUT(movingCount, 0, CH(task_resetStats, task_stats));
    CHAN_OUT(stationaryCount, 0, CH(task_resetStats, task_stats));
    CHAN_OUT(totalCount, 0, CH(task_resetStats, task_stats));

    CHAN_OUT(samplesInWindow, 0, CH(task_resetStats, task_sample));

    TRANSITION_TO(task_sample);
}

void task_sample()
{
    accelReading sample;
    unsigned samplesInWindow;

    printf("sample\r\n");

#ifdef SHOW_PROGRESS_ON_LEDS
    blink(SAMPLE_BLINKS, SAMPLE_BLINK_DURATION, LED1 | LED2);
#endif

    ACCEL_singleSample(&sample);

    samplesInWindow = *CHAN_IN3(samplesInWindow,
                               CH(task_resetStats, task_sample),
                               CH(task_selectMode, task_sample),
                               SELF_IN_CH(task_sample));

    CHAN_OUT(window[samplesInWindow], sample,
             MC_OUT_CH(ch_sample_window, task_sample, task_transform, task_featurize));
    samplesInWindow++;
    printf("sample: sample %u %u %u window %u\r\n",
           sample.x, sample.y, sample.z, samplesInWindow);

    if (samplesInWindow < ACCEL_WINDOW_SIZE) {
        CHAN_OUT(samplesInWindow, samplesInWindow, SELF_OUT_CH(task_sample));
        TRANSITION_TO(task_sample);
    } else {
        CHAN_OUT(samplesInWindow, 0, SELF_OUT_CH(task_sample));
        TRANSITION_TO(task_transform);
    }
}

void task_transform()
{
    unsigned i;

    printf("transform\r\n");

    accelReading *sample;
    for (i = 0; i < ACCEL_WINDOW_SIZE; i++) {
        sample = CHAN_IN1(window[i], MC_IN_CH(ch_sample_window, task_sample, task_transform));
        if (sample->x < SAMPLE_NOISE_FLOOR)
            sample->x = 0;
        if (sample->y < SAMPLE_NOISE_FLOOR)
            sample->y = 0;
        if (sample->z < SAMPLE_NOISE_FLOOR)
            sample->z = 0;

        CHAN_OUT(window[i], *sample, CH(task_transform, task_featurize));
    }
    TRANSITION_TO(task_featurize);
}

void task_featurize()
{
   accelReading *reading;
   accelReading mean, stddev;
   features_t features;
   run_mode_t mode;

   printf("featurize\r\n");

#ifdef SHOW_PROGRESS_ON_LEDS
   blink(FEATURIZE_BLINKS, FEATURIZE_BLINK_DURATION, LED1 | LED2);
#endif
 
   mean.x = mean.y = mean.z = 0;
   stddev.x = stddev.y = stddev.z = 0;
   int i;
   for (i = 0; i < ACCEL_WINDOW_SIZE; i++) {
       reading = CHAN_IN2(window[i], MC_IN_CH(ch_sample_window, task_sample, task_featurize),
                                    CH(task_transform, task_featurize));
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
       reading = CHAN_IN2(window[i], MC_IN_CH(ch_sample_window, task_sample, task_featurize),
                                    CH(task_transform, task_featurize));
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

   printf("featurize: features: mean %u stddev %u\r\n",
           features.meanmag, features.stddevmag);

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

   printf("classify\r\n");
  
    features = *CHAN_IN1(features, CH(task_featurize, task_classify));

    // TODO: does it make sense to get a reference to the whole model at once?

    // TODO: use features obj directly
    meanmag = features.meanmag;
    stddevmag = features.stddevmag;
  
    for (i = 0; i < MODEL_COMPARISONS; i += NUM_FEATURES) {
        model_features = *CHAN_IN2(model_stationary[i], CH(task_init, task_classify),
                                                    CH(task_train, task_classify));
        long int stat_mean_err = (model_features.meanmag > meanmag)
            ? (model_features.meanmag - meanmag)
            : (meanmag - model_features.meanmag);

        long int stat_sd_err = (model_features.stddevmag > stddevmag)
            ? (model_features.stddevmag - stddevmag)
            : (stddevmag - model_features.stddevmag);

        model_features = *CHAN_IN2(model_moving[i], CH(task_init, task_classify),
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

    printf("classify: class 0x%x\r\n", class);
  
    TRANSITION_TO(task_stats);
}

void task_stats()
{
    unsigned totalCount = 0, movingCount = 0, stationaryCount = 0;
    class_t class;

    printf("stats\r\n");

    totalCount = *CHAN_IN2(totalCount, CH(task_resetStats, task_stats),
                                      SELF_IN_CH(task_stats));

    totalCount++;
    printf("stats: total %u\r\n", totalCount);

    CHAN_OUT(totalCount, totalCount, SELF_OUT_CH(task_stats));

    class = *CHAN_IN1(class, CH(task_classify, task_stats));

    switch (class) {
        case CLASS_MOVING:

#if defined (SHOW_RESULT_ON_LEDS)
            blink(CLASSIFY_BLINKS, CLASSIFY_BLINK_DURATION, LED1);
#endif //SHOW_RESULT_ON_LEDS

            movingCount = *CHAN_IN2(movingCount, CH(task_resetStats, task_stats),
                                                SELF_IN_CH(task_stats));
            movingCount++;
            printf("stats: moving %u\r\n", movingCount);
            CHAN_OUT(movingCount, movingCount, SELF_OUT_CH(task_stats));
            break;
        case CLASS_STATIONARY:

#if defined (SHOW_RESULT_ON_LEDS)
            blink(CLASSIFY_BLINKS, CLASSIFY_BLINK_DURATION, LED2);
#endif //SHOW_RESULT_ON_LEDS

            stationaryCount = *CHAN_IN2(stationaryCount, CH(task_resetStats, task_stats),
                                                      SELF_IN_CH(task_stats));
            stationaryCount++;
            printf("stats: stationary %u\r\n", stationaryCount);
            CHAN_OUT(stationaryCount, stationaryCount, SELF_OUT_CH(task_stats));
            break;
    }

    if (totalCount > SAMPLES_TO_COLLECT) {

        // Get the other count from the channel: this only happens once per
        // acquisition run: we're saving 50% reads all other times.
        switch (class) {
            case CLASS_MOVING:
                stationaryCount = *CHAN_IN2(stationaryCount,
                                            CH(task_resetStats, task_stats),
                                            SELF_IN_CH(task_stats));
                break;
            case CLASS_STATIONARY:
                movingCount = *CHAN_IN2(movingCount,
                                        CH(task_resetStats, task_stats),
                                        SELF_IN_CH(task_stats));
                break;
        }


        // This is "I/O" (specifically, output). Yes, it's to non-volatile
        // memory, which may seem confusing, but for this program to have any
        // purpose, there needs to be I/O of the computed results to somewhere.
        // In channels model, this simply becomes explicit, because channel
        // non-volatile memory space has nothing to do with the memory space
        // where the results happen to be output.
        resultStationaryPct = ((float)stationaryCount / (float)totalCount) * 100.0f;
        resultMovingPct = ((float)movingCount / (float)totalCount) * 100.0f;

        printf("stats: stat %u/%u (%u%%) moving %u/%u (%u%%)\r\n",
               stationaryCount, totalCount, (unsigned)resultStationaryPct,
               movingCount, totalCount, (unsigned)resultMovingPct);

#if defined (SHOW_RESULT_ON_LEDS)
        P4OUT &= ~PIN_LED2;
        PJOUT &= ~PIN_LED1;
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

    printf("warmup\r\n");

#ifdef SHOW_PROGRESS_ON_LEDS
    blink(WARMUP_BLINKS, WARMUP_BLINK_DURATION, LED1 | LED2);
#endif

    discardedSamplesCount = *CHAN_IN2(discardedSamplesCount,
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
        printf("warmup: discarded %u\r\n", discardedSamplesCount);
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

    printf("train\r\n");

#ifdef SHOW_PROGRESS_ON_LEDS
    blink(TRAIN_BLINKS, TRAIN_BLINK_DURATION, LED1 | LED2);
#endif

    features = *CHAN_IN1(features, CH(task_featurize, task_train));
    trainingSetSize = *CHAN_IN2(trainingSetSize, CH(task_warmup, task_train),
                                                SELF_IN_CH(task_train));
    class = *CHAN_IN1(class, CH(task_selectMode, task_train));

    switch (class) {
        case CLASS_STATIONARY:
            CHAN_OUT(model_stationary[trainingSetSize], features, CH(task_train, task_classify));
            break;
        case CLASS_MOVING:
            CHAN_OUT(model_moving[trainingSetSize], features, CH(task_train, task_classify));
            break;
    }

    trainingSetSize++;
    printf("train: class %u count %u/%u\r\n", class,
           trainingSetSize, TRAINING_SET_SIZE);
    CHAN_OUT(trainingSetSize, trainingSetSize, SELF_IN_CH(task_train));

    if (trainingSetSize < TRAINING_SET_SIZE)
        TRANSITION_TO(task_sample);
    else
        TRANSITION_TO(task_idle);
}

void task_idle() {
#ifdef SHOW_PROGRESS_ON_LEDS
    blink(IDLE_BLINKS, IDLE_BLINK_DURATION, LED1 | LED2);
#endif
    delay(IDLE_WAIT);

    printf("idle\r\n");

    TRANSITION_TO(task_selectMode);
}

INIT_FUNC(initializeHardware)
ENTRY_TASK(task_init)
