EXEC = ar.out

OBJECTS = \
	main.o \

LIBCHAIN_ENABLE_DIAGNOSTICS ?= 0
CONFIG_PRINTF_LIB ?= libmspconsole

include $(MAKER_ROOT)/Makefile.gcc

CFLAGS += \
	-I $(LIBCHAIN_ROOT)/src/include \
	-I $(LIBEDB_ROOT)/src/include \
	-I $(WISP_BASE_ROOT) \
	-I $(WISP_BASE_ROOT)/Sensors \
	-I $(WISP_BASE_ROOT)/wired \

ifeq ($(LIBCHAIN_ENABLE_DIAGNOSTICS),1)
CFLAGS += -DLIBCHAIN_ENABLE_DIAGNOSTICS
endif

ifeq ($(CONFIG_PRINTF_LIB),libedb)
CFLAGS += -DCONFIG_LIBEDB_PRINTF
LIBS += -ledb
LFLAGS += -L$(LIBEDB_ROOT)/gcc
endif

ifeq ($(CONFIG_PRINTF_LIB),libmspconsole)
CFLAGS += -DCONFIG_LIBMSPCONSOLE_PRINTF
LIBS += -lmspconsole
LFLAGS += -L$(LIBMSPCONSOLE_ROOT)/bld/gcc
endif

LFLAGS += \
	-L $(LIBCHAIN_ROOT)/bld/gcc \
	-L $(WISP_BASE_ROOT)/gcc \

LIBS += \
    -lchain \
	-lwisp-base \
	-lm \

VPATH = ../../src
