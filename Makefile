CC              := g++
CFLAGS          := -Wall
CPPFLAGS        := -I./ -I/usr/X11R6/include/Xm -I/usr/X11R6/include -I/opt/homebrew/Cellar/openmotif/2.3.8_2/include -I/opt/homebrew/include -Wno-error=address -Wno-pointer-to-int-cast -I. -fpermissive

LDFLAGS         := -L/usr/lib/X11R6 -lXm -lXaw -lXmu -lXt -lX11 -lpthread
#LDFLAGS         := -L/usr/X11R6/lib -L /usr/X11R6/LessTif/Motif1.2/lib -L /opt/homebrew/lib -lXm -lXaw -lXmu -lXt -lX11 -lICE -lSM

# Uncomment this next line if you'd like to compile the graphical version of
# the checkers server.
#CFLAGS          += -DGRAPHICS

all: checkers computer nn dumper
checkers: graphics.o
#computer: myprog.o
computer: player.c
	${CC} ${cppflags} ${cflags} player.c playerHelper.c -o computer
nn: nn.cpp
	${CC} ${CPPFLAGS} ${CFLAGS} nn.cpp -o nn
dumper: player_dumper.c
	${CC} ${cppflags} ${cflags} player_dumper.c playerHelper.c -o dumper

.PHONY: clean
clean:	
	@-rm checkers computer *.o dumper
