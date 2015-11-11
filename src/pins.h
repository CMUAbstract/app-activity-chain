#ifndef PIN_ASSIGN_H
#define PIN_ASSIGN_H

// Ugly workaround to make the pretty GPIO macro work for OUT register
// (a control bit for TAxCCTLx uses the name 'OUT')
#undef OUT

#define BIT_INNER(idx) BIT ## idx
#define BIT(idx) BIT_INNER(idx)

#define GPIO_INNER(port, reg) P ## port ## reg
#define GPIO(port, reg) GPIO_INNER(port, reg)

#if defined(BOARD_WISP)
#define     PORT_LED_1           4
#define     PIN_LED_1            0
#define     PORT_LED_2           J
#define     PIN_LED_2            6

#define     PORT_AUX            3
#define     PIN_AUX_1           4
#define     PIN_AUX_2           5


#define     PORT_AUX3           1
#define     PIN_AUX_3           4

#elif defined(BOARD_MSP_TS430)

#define     PORT_LED_1           1
#define     PIN_LED_1            1
#define     PORT_LED_2           1
#define     PIN_LED_2            2
#define     PORT_LED_3           1
#define     PIN_LED_3            0

#define     PORT_AUX            3
#define     PIN_AUX_1           4
#define     PIN_AUX_2           5


#define     PORT_AUX3           1
#define     PIN_AUX_3           4

#endif // BOARD_*

#endif
