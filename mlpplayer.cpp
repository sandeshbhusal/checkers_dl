#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/times.h>
#include <time.h>
#include <float.h>
#include "myprog.h"
#include "defs.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace std;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;
using namespace tiny_dnn;

network<sequential> net;
int randomprob = 0.2;

/* copy numbytes from src to destination
   before the copy happens the destSize bytes of the destination array is set to 0s
*/
void safeCopy(char *dest, char *src, int destSize, int numbytes)
{
    memset(dest, 0, destSize);
    memcpy(dest, src, numbytes);
}

void printBoard(struct State *state)
{
    int y, x;

    for (y = 0; y < 8; y++)
    {
        for (x = 0; x < 8; x++)
        {
            if (x % 2 != y % 2)
            {
                if (empty(state->board[y][x]))
                {
                    fprintf(stderr, " ");
                }
                else if (king(state->board[y][x]))
                {
                    if (color(state->board[y][x]) == 2)
                        fprintf(stderr, "B");
                    else
                        fprintf(stderr, "A");
                }
                else if (piece(state->board[y][x]))
                {
                    if (color(state->board[y][x]) == 2)
                        fprintf(stderr, "b");
                    else
                        fprintf(stderr, "a");
                }
            }
            else
            {
                fprintf(stderr, " ");
            }
        }
        fprintf(stderr, "\n");
    }
}

vec_t generate_input_from_board(int player, struct State *state)
{
    printBoard(state);

    vec_t rval;
    rval.push_back(player);

    for (int x = 0; x < 8; x++)
    {
        for (int y = 0; y < 8; y++)
        {
            if (x % 2 != y % 2)
            {
                char ch = state->board[y][x];
                if (empty(state->board[y][x]))
                {
                    rval.push_back(0.0);
                }
                else if (king(state->board[y][x]))
                {
                    if (color(ch) == player)
                        rval.push_back(2.0);
                    else
                        rval.push_back(-2.0);
                }
                else if (piece(state->board[y][x]))
                {
                    if (color(ch) == player)
                        rval.push_back(1.0);
                    else
                        rval.push_back(-1.0);
                }
            }
        }
    }

    return rval;
}

/* Employ your favorite search to find the best move here.  */
/* This example code shows you how to call the FindLegalMoves function */
/* and the PerformMove function */
void FindBestMove(int player, char board[8][8], char *bestmove)
{
    net.load("testnet");

    int bestMoveIndex;
    double bestMoveEval = -DBL_MAX;
    struct State state;

    setupBoardState(&state, player, board);

    printBoard(&state);
    // Here's an example loop walking through all the moves in the board state
    for (int i = 0; i < state.numLegalMoves; i++)
    {
        State nextState;
        memcpy(&nextState, &state, sizeof(State));
        performMove(&nextState, i);
        vec_t inputs = generate_input_from_board(player, &nextState);
        float value = net.predict(inputs)[0];
        if (value > bestMoveEval){
            bestMoveEval = value;
            bestMoveIndex = i;
        }
    }

    safeCopy(bestmove, state.movelist[bestMoveIndex], MaxMoveLength, MoveLength(state.movelist[bestMoveIndex]));
}
