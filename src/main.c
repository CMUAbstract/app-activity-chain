#include <msp430.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <math.h>

#include <wisp-base.h>
#include <accel.h>
#include <spi.h>

#include <msp-builtins.h>
#include <msp-math.h>

#include <libmsp/mem.h>
#include <libio/log.h>
#include <libchain/chain.h>

#ifdef CONFIG_LIBEDB_PRINTF
#include <libedb/edb.h>
#endif

#include "pins.h"

// #define SHOW_RESULT_ON_LEDS
// #define SHOW_PROGRESS_ON_LEDS
// #define SHOW_BOOT_ON_LEDS

// Number of samples to discard before recording training set
#define NUM_WARMUP_SAMPLES 3

#define ACCEL_WINDOW_SIZE 3
#define MODEL_SIZE 16
#define SAMPLE_NOISE_FLOOR 10 // TODO: made up value

// Number of classifications to complete in one experiment
#define SAMPLES_TO_COLLECT 128

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
    MODE_IDLE = (BIT(PIN_AUX_1) | BIT(PIN_AUX_2)),
    MODE_TRAIN_STATIONARY = BIT(PIN_AUX_1),
    MODE_TRAIN_MOVING = BIT(PIN_AUX_2),
    MODE_RECOGNIZE = 0, // default
} run_mode_t;

struct msg_stats {
    CHAN_FIELD(unsigned, totalCount);
    CHAN_FIELD(unsigned, movingCount);
    CHAN_FIELD(unsigned, stationaryCount);
};

struct msg_self_stats {
    SELF_CHAN_FIELD(unsigned, totalCount);
    SELF_CHAN_FIELD(unsigned, movingCount);
    SELF_CHAN_FIELD(unsigned, stationaryCount);
};
#define FIELD_INIT_msg_self_stats {\
    SELF_FIELD_INITIALIZER, \
    SELF_FIELD_INITIALIZER, \
    SELF_FIELD_INITIALIZER, \
}

struct msg_train {
    CHAN_FIELD(unsigned, trainingSetSize);
};

struct msg_self_train {
    SELF_CHAN_FIELD(unsigned, trainingSetSize);
};
#define FIELD_INIT_msg_self_train {\
    SELF_FIELD_INITIALIZER \
}

struct msg_windowSize {
    CHAN_FIELD(unsigned, samplesInWindow);
};

struct msg_self_windowSize {
    SELF_CHAN_FIELD(unsigned, samplesInWindow);
};
#define FIELD_INIT_msg_self_windowSize {\
    SELF_FIELD_INITIALIZER \
}

struct msg_warmup {
    CHAN_FIELD(unsigned, discardedSamplesCount);
};

struct msg_self_warmup {
    SELF_CHAN_FIELD(unsigned, discardedSamplesCount);
};
#define FIELD_INIT_msg_self_warmup {\
    SELF_FIELD_INITIALIZER \
}

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

struct msg_pinState {
    CHAN_FIELD(uint8_t, pin_state);
};

struct msg_self_pinState {
    SELF_CHAN_FIELD(uint8_t, pin_state);
};
#define FIELD_INIT_msg_self_pinState {\
    SELF_FIELD_INITIALIZER \
}

TASK(1, task_init)
TASK(2, task_selectMode)
TASK(3, task_resetStats)
TASK(4, task_sample)
TASK(5, task_transform)
TASK(6, task_featurize)
TASK(7, task_classify)
TASK(8, task_stats)
TASK(9, task_warmup)
TASK(10, task_train)
TASK(11, task_idle)

CHANNEL(task_init, task_classify, msg_model);
CHANNEL(task_init, task_selectMode, msg_pinState);

CHANNEL(task_selectMode, task_warmup, msg_warmup);
CHANNEL(task_selectMode, task_featurize, msg_mode);
CHANNEL(task_selectMode, task_train, msg_class);
CHANNEL(task_selectMode, task_sample, msg_windowSize);
SELF_CHANNEL(task_selectMode, msg_self_pinState);

CHANNEL(task_resetStats, task_stats, msg_stats);
CHANNEL(task_resetStats, task_sample, msg_windowSize);

MULTICAST_CHANNEL(msg_window, ch_sample_window,
                 task_sample, task_transform, task_featurize);
SELF_CHANNEL(task_sample, msg_self_windowSize);
CHANNEL(task_transform, task_featurize, msg_window);

CHANNEL(task_featurize, task_train, msg_features);
CHANNEL(task_featurize, task_classify, msg_features);

CHANNEL(task_classify, task_stats, msg_class);

SELF_CHANNEL(task_stats, msg_self_stats);

SELF_CHANNEL(task_warmup, msg_self_warmup);
CHANNEL(task_warmup, task_train, msg_train);

SELF_CHANNEL(task_train, msg_self_train);
CHANNEL(task_train, task_classify, msg_model);

#if defined(SHOW_RESULT_ON_LEDS) || defined(SHOW_PROGRESS_ON_LEDS)
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
#endif

void initializeHardware()
{
    threeAxis_t_8 accelID = {0};

    WISP_init();

#ifdef CONFIG_EDB
    debug_setup();
#endif

    INIT_CONSOLE();

    GPIO(PORT_LED_1, DIR) |= BIT(PIN_LED_1);
    GPIO(PORT_LED_2, DIR) |= BIT(PIN_LED_2);
#if defined(PORT_LED_3) // inidicates power-on when available
    GPIO(PORT_LED_3, DIR) |= BIT(PIN_LED_3);
    GPIO(PORT_LED_3, OUT) |= BIT(PIN_LED_3);
#endif

#ifdef SHOW_BOOT_ON_LEDS
    GPIO(PORT_LED_1, OUT) |= BIT(PIN_LED_1);
    GPIO(PORT_LED_2, OUT) |= BIT(PIN_LED_2);
    delay(SEC_TO_CYCLES * 5);
    GPIO(PORT_LED_1, OUT) &= ~BIT(PIN_LED_1);
    GPIO(PORT_LED_2, OUT) &= ~BIT(PIN_LED_2);
#endif

    __enable_interrupt();

    LOG("init: initializing accel\r\n");

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
    BITSET(P4OUT, PIN_ACCEL_EN);

    BITSET(P2SEL1, PIN_ACCEL_SCLK | PIN_ACCEL_MISO | PIN_ACCEL_MOSI);
    BITCLR(P2SEL0, PIN_ACCEL_SCLK | PIN_ACCEL_MISO | PIN_ACCEL_MOSI);
    __delay_cycles(1000);
    SPI_initialize();
    __delay_cycles(1000);
    ACCEL_range();
    __delay_cycles(1000);
    ACCEL_initialize();
    __delay_cycles(1000);
    ACCEL_readID(&accelID);

    LOG("init: accel hw id: 0x%x\r\n", accelID.x);

    PRINTF(".%u.\r\n", curctx->task->idx);
}

void task_init()
{
    LOG("init\r\n");

    uint8_t pin_state = MODE_IDLE;
    CHAN_OUT1(uint8_t, pin_state, pin_state, CH(task_init, task_selectMode));

    TRANSITION_TO(task_selectMode);
}
void task_selectMode()
{
    const unsigned zero = 0;

    run_mode_t mode;
    class_t class;

#ifdef SHOW_PROGRESS_ON_LEDS
    blink(SELECT_MODE_BLINKS, SELECT_MODE_BLINK_DURATION, LED1 | LED2);
#endif

    uint8_t pin_state = GPIO(PORT_AUX, IN) & (BIT(PIN_AUX_1) | BIT(PIN_AUX_2));

    // Don't re-launch training after finishing training
    uint8_t prev_pin_state = *CHAN_IN2(uint8_t, pin_state,
                                       CH(task_init, task_selectMode),
                                       SELF_IN_CH(task_selectMode));
    if ((pin_state == MODE_TRAIN_STATIONARY ||
        pin_state == MODE_TRAIN_MOVING) &&
        pin_state == prev_pin_state) {
        pin_state = MODE_IDLE;
    } else {
        CHAN_OUT1(uint8_t, pin_state, pin_state, SELF_OUT_CH(task_selectMode));
    }

    LOG("selectMode: 0x%x\r\n", pin_state);

    switch(pin_state) {
        case MODE_TRAIN_STATIONARY:
            CHAN_OUT1(unsigned, discardedSamplesCount, zero,
                      CH(task_selectMode, task_warmup));
            mode = MODE_TRAIN_STATIONARY;
            CHAN_OUT1(run_mode_t, mode, mode,
                      CH(task_selectMode, task_featurize));
            class = CLASS_STATIONARY;
            CHAN_OUT1(class_t, class, class,
                      CH(task_selectMode, task_train));
            CHAN_OUT1(unsigned, samplesInWindow, zero,
                      CH(task_selectMode, task_sample));

            TRANSITION_TO(task_warmup);
            break;

        case MODE_TRAIN_MOVING:
            CHAN_OUT1(unsigned, discardedSamplesCount, zero,
                     CH(task_selectMode, task_warmup));
            mode = MODE_TRAIN_MOVING;
            CHAN_OUT1(run_mode_t, mode, mode,
                     CH(task_selectMode, task_featurize));
            class = CLASS_MOVING;
            CHAN_OUT1(class_t, class, class,
                     CH(task_selectMode, task_train));
            CHAN_OUT1(unsigned, samplesInWindow, zero,
                     CH(task_selectMode, task_sample));

            TRANSITION_TO(task_warmup);
            break;

        case MODE_RECOGNIZE:
            mode = MODE_RECOGNIZE;
            CHAN_OUT1(run_mode_t, mode, mode,
                      CH(task_selectMode, task_featurize));

            TRANSITION_TO(task_resetStats);
            break;

        default:
            TRANSITION_TO(task_idle);
    }
}

void task_resetStats()
{
    const unsigned zero = 0;

    // NOTE: could roll this into selectMode task, but no compelling reason

    LOG("resetStats\r\n");

    // NOTE: not combined into one struct because not all code paths use both
    CHAN_OUT1(unsigned, movingCount, zero, CH(task_resetStats, task_stats));
    CHAN_OUT1(unsigned, stationaryCount, zero, CH(task_resetStats, task_stats));
    CHAN_OUT1(unsigned, totalCount, zero, CH(task_resetStats, task_stats));

    CHAN_OUT1(unsigned, samplesInWindow, zero, CH(task_resetStats, task_sample));

    TRANSITION_TO(task_sample);
}

void task_sample()
{
    accelReading sample;
    unsigned samplesInWindow;

    LOG("sample\r\n");

#ifdef SHOW_PROGRESS_ON_LEDS
    blink(SAMPLE_BLINKS, SAMPLE_BLINK_DURATION, LED1 | LED2);
#endif

    ACCEL_singleSample(&sample);

    samplesInWindow = *CHAN_IN3(unsigned, samplesInWindow,
                               CH(task_resetStats, task_sample),
                               CH(task_selectMode, task_sample),
                               SELF_IN_CH(task_sample));

    CHAN_OUT1(accelReading, window[samplesInWindow], sample,
             MC_OUT_CH(ch_sample_window, task_sample,
                       task_transform, task_featurize));
    samplesInWindow++;
    LOG("sample: sample %u %u %u window %u\r\n",
           sample.x, sample.y, sample.z, samplesInWindow);

    if (samplesInWindow < ACCEL_WINDOW_SIZE) {
        CHAN_OUT1(unsigned, samplesInWindow, samplesInWindow,
                  SELF_OUT_CH(task_sample));
        TRANSITION_TO(task_sample);
    } else {
        const unsigned zero = 0;
        CHAN_OUT1(unsigned, samplesInWindow, zero, SELF_OUT_CH(task_sample));
        TRANSITION_TO(task_transform);
    }
}

void task_transform()
{
    unsigned i;

    LOG("transform\r\n");

    accelReading *sample;
    accelReading transformedSample;

    for (i = 0; i < ACCEL_WINDOW_SIZE; i++) {
        sample = CHAN_IN1(accelReading, window[i],
                          MC_IN_CH(ch_sample_window,
                                   task_sample, task_transform));

        if (sample->x < SAMPLE_NOISE_FLOOR ||
            sample->y < SAMPLE_NOISE_FLOOR ||
            sample->z < SAMPLE_NOISE_FLOOR) {

            transformedSample.x = (sample->x > SAMPLE_NOISE_FLOOR)
                                  ? sample->x : 0;
            transformedSample.y = (sample->y > SAMPLE_NOISE_FLOOR)
                                  ? sample->y : 0;
            transformedSample.z = (sample->z > SAMPLE_NOISE_FLOOR)
                                   ? sample->z : 0;

            CHAN_OUT1(accelReading, window[i], transformedSample,
                      CH(task_transform, task_featurize));
        }
    }
    TRANSITION_TO(task_featurize);
}

void task_featurize()
{
   accelReading *reading;
   accelReading mean, stddev;
   features_t features;
   run_mode_t mode;

   LOG("featurize\r\n");

#ifdef SHOW_PROGRESS_ON_LEDS
   blink(FEATURIZE_BLINKS, FEATURIZE_BLINK_DURATION, LED1 | LED2);
#endif
 
   mean.x = mean.y = mean.z = 0;
   stddev.x = stddev.y = stddev.z = 0;
   int i;
   for (i = 0; i < ACCEL_WINDOW_SIZE; i++) {
       reading = CHAN_IN2(accelReading, window[i],
                          MC_IN_CH(ch_sample_window, task_sample, task_featurize),
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
       reading = CHAN_IN2(accelReading, window[i],
                          MC_IN_CH(ch_sample_window, task_sample, task_featurize),
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
 
    unsigned meanmag = mean.x*mean.x + mean.y*mean.y + mean.z*mean.z;
    unsigned stddevmag = stddev.x*stddev.x + stddev.y*stddev.y + stddev.z*stddev.z;

    features.meanmag   = sqrt16(meanmag);
    features.stddevmag = sqrt16(stddevmag);
 
   mode = *CHAN_IN1(run_mode_t, mode, CH(task_selectMode, task_featurize));

   LOG("featurize: features: mean %u stddev %u\r\n",
           features.meanmag, features.stddevmag);

   switch (mode) {
       case MODE_TRAIN_STATIONARY:
       case MODE_TRAIN_MOVING:
           CHAN_OUT1(features_t, features, features,
                     CH(task_featurize, task_train));
           TRANSITION_TO(task_train);
           break;
        case MODE_RECOGNIZE:
           CHAN_OUT1(features_t, features, features,
                     CH(task_featurize, task_classify));
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

   LOG("classify\r\n");
  
    features = *CHAN_IN1(features_t, features,
                         CH(task_featurize, task_classify));

    // TODO: does it make sense to get a reference to the whole model at once?

    // TODO: use features obj directly
    meanmag = features.meanmag;
    stddevmag = features.stddevmag;
  
    for (i = 0; i < MODEL_SIZE; ++i) {
        model_features = *CHAN_IN2(features_t, model_stationary[i],
                                   CH(task_init, task_classify),
                                   CH(task_train, task_classify));
        long int stat_mean_err = (model_features.meanmag > meanmag)
            ? (model_features.meanmag - meanmag)
            : (meanmag - model_features.meanmag);

        long int stat_sd_err = (model_features.stddevmag > stddevmag)
            ? (model_features.stddevmag - stddevmag)
            : (stddevmag - model_features.stddevmag);

        model_features = *CHAN_IN2(features_t, model_moving[i],
                                   CH(task_init, task_classify),
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
    CHAN_OUT1(class_t, class, class, CH(task_classify, task_stats));

    LOG("classify: class 0x%x\r\n", class);
  
    TRANSITION_TO(task_stats);
}

void task_stats()
{
    unsigned totalCount = 0, movingCount = 0, stationaryCount = 0;
    class_t class;

    LOG("stats\r\n");

    totalCount = *CHAN_IN2(unsigned, totalCount,
                           CH(task_resetStats, task_stats),
                           SELF_IN_CH(task_stats));

    totalCount++;
    LOG("stats: total %u\r\n", totalCount);

    CHAN_OUT1(unsigned, totalCount, totalCount, SELF_OUT_CH(task_stats));

    class = *CHAN_IN1(class_t, class, CH(task_classify, task_stats));

    switch (class) {
        case CLASS_MOVING:

#if defined (SHOW_RESULT_ON_LEDS)
            blink(CLASSIFY_BLINKS, CLASSIFY_BLINK_DURATION, LED1);
#endif //SHOW_RESULT_ON_LEDS

            movingCount = *CHAN_IN2(unsigned, movingCount,
                                    CH(task_resetStats, task_stats),
                                    SELF_IN_CH(task_stats));
            movingCount++;
            LOG("stats: moving %u\r\n", movingCount);
            CHAN_OUT1(unsigned, movingCount, movingCount,
                      SELF_OUT_CH(task_stats));

            break;
        case CLASS_STATIONARY:

#if defined (SHOW_RESULT_ON_LEDS)
            blink(CLASSIFY_BLINKS, CLASSIFY_BLINK_DURATION, LED2);
#endif //SHOW_RESULT_ON_LEDS

            stationaryCount = *CHAN_IN2(unsigned, stationaryCount,
                                        CH(task_resetStats, task_stats),
                                        SELF_IN_CH(task_stats));
            stationaryCount++;
            LOG("stats: stationary %u\r\n", stationaryCount);
            CHAN_OUT1(unsigned, stationaryCount, stationaryCount,
                      SELF_OUT_CH(task_stats));

            break;
    }

    if (totalCount == SAMPLES_TO_COLLECT) {

        // Get the other count from the channel: this only happens once per
        // acquisition run: we're saving 50% reads all other times.
        switch (class) {
            case CLASS_MOVING:
                stationaryCount = *CHAN_IN2(unsigned, stationaryCount,
                                            CH(task_resetStats, task_stats),
                                            SELF_IN_CH(task_stats));
                break;
            case CLASS_STATIONARY:
                movingCount = *CHAN_IN2(unsigned, movingCount,
                                        CH(task_resetStats, task_stats),
                                        SELF_IN_CH(task_stats));
                break;
        }


        unsigned resultStationaryPct = stationaryCount * 100 / totalCount;
        unsigned resultMovingPct = movingCount * 100 / totalCount;

        unsigned sum = stationaryCount + movingCount;
        PRINTF("stats: s %u (%u%%) m %u (%u%%) sum/tot %u/%u: %c\r\n",
               stationaryCount, resultStationaryPct,
               movingCount, resultMovingPct,
               totalCount, sum, sum == totalCount ? 'V' : 'X');

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

    LOG("warmup\r\n");

#ifdef SHOW_PROGRESS_ON_LEDS
    blink(WARMUP_BLINKS, WARMUP_BLINK_DURATION, LED1 | LED2);
#endif

    discardedSamplesCount = *CHAN_IN2(unsigned, discardedSamplesCount,
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
        LOG("warmup: discarded %u\r\n", discardedSamplesCount);
        CHAN_OUT1(unsigned, discardedSamplesCount, discardedSamplesCount,
                  SELF_OUT_CH(task_warmup));
        TRANSITION_TO(task_warmup);
    } else {
        const unsigned zero = 0;
        CHAN_OUT1(unsigned, trainingSetSize, zero, CH(task_warmup, task_train));
        TRANSITION_TO(task_sample);
    }
}

void task_train()
{
    features_t features;
    unsigned trainingSetSize;;
    unsigned class;

    LOG("train\r\n");

#ifdef SHOW_PROGRESS_ON_LEDS
    blink(TRAIN_BLINKS, TRAIN_BLINK_DURATION, LED1 | LED2);
#endif

    features = *CHAN_IN1(features_t, features, CH(task_featurize, task_train));
    trainingSetSize = *CHAN_IN2(unsigned, trainingSetSize,
                                CH(task_warmup, task_train),
                                SELF_IN_CH(task_train));
    class = *CHAN_IN1(class_t, class, CH(task_selectMode, task_train));

    switch (class) {
        case CLASS_STATIONARY:
            CHAN_OUT1(features_t, model_stationary[trainingSetSize], features,
                      CH(task_train, task_classify));
            break;
        case CLASS_MOVING:
            CHAN_OUT1(features_t, model_moving[trainingSetSize], features,
                      CH(task_train, task_classify));
            break;
    }

    trainingSetSize++;
    LOG("train: class %u count %u/%u\r\n", class,
           trainingSetSize, MODEL_SIZE);
    CHAN_OUT1(unsigned, trainingSetSize, trainingSetSize,
              SELF_IN_CH(task_train));

    if (trainingSetSize < MODEL_SIZE) {
        TRANSITION_TO(task_sample);
    } else {
        PRINTF("train: class %u done (mn %u sd %u)\r\n",
               class, features.meanmag, features.stddevmag);
        TRANSITION_TO(task_idle);
    }
}

void task_idle() {
#ifdef SHOW_PROGRESS_ON_LEDS
    blink(IDLE_BLINKS, IDLE_BLINK_DURATION, LED1 | LED2);
#endif
#if CONT_POWER
    delay(IDLE_WAIT);
#endif

    LOG("idle\r\n");

    TRANSITION_TO(task_selectMode);
}

INIT_FUNC(initializeHardware)
ENTRY_TASK(task_init)
