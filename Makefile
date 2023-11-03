CC              := g++
CFLAGS          := -Wall
CPPFLAGS        := -I./ -I/usr/X11R6/include/Xm -I/usr/X11R6/include -I/opt/homebrew/Cellar/openmotif/2.3.8_2/include -I/opt/homebrew/include -Wno-error=address -Wno-pointer-to-int-cast -I. -fpermissive

LDFLAGS         := -L/usr/lib/X11R6 -lXm -lXaw -lXmu -lXt -lX11 -lpthread

# Uncomment this next line if you'd like to compile the graphical version of
# the checkers server.
#CFLAGS          += -DGRAPHICS

all: checkers computer nn selflearncheckers ai

# Normal checkers server.
checkers: graphics.o

# Self-learning checkers server - updates the neural net directly from players' boards' history.
selflearncheckers: selflearncheckers.cpp graphics.o
	${CC} ${cppflags} ${cflags} ${LDFLAGS} selflearncheckers.cpp graphics.o -o selflearncheckers -lpthread -I.

# Random player
computer: player.c
	${CC} ${cppflags} ${CFLAGS} player.c playerHelper.c -o computer

# Neural network trainer - trains the neural net based on external data.csv files.
nn: nn.cpp
	${CC} ${CPPFLAGS} ${CFLAGS} nn.cpp -o trainer -lpthread -I.

# MCTS agent with selfplay based on the neural network state-value prediction.
ai: mlpcts.cpp
	${CC} ${cppflags} ${cflags} mlpcts.cpp -o ai -I.

.PHONY: clean
clean:	
	@-rm checkers computer *.o trainer mlpplayer selflearncheckers nn ai checkers_display
