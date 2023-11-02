#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <float.h>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/times.h>
#include "myprog.h"
#include <time.h>
#include "tiny_dnn/tiny_dnn.h"

using namespace std;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;
using namespace tiny_dnn;

network<sequential> net;

float get_board_eval(struct State *state)
{
    // Given a board state, convert it to a vec_t, and gather the predicted value for this state.
    float rval = 0.0;

    vec_t pred_input;

    // All calculations are done in reference to the parent player.
    int parent_player = state->player % 2 + 1;
    pred_input.push_back(parent_player);

    std::cerr << "Parent player is " << parent_player<< endl;
    for (int y = 0; y < 8; y++)
    {
        for (int x = 0; x < 8; x++)
        {
            if (y % 2 != x % 2)
            {
                char ch = state->board[y][x];
                if (ch == ' ')
                    pred_input.push_back(0.0);

                if (ch == 'a')
                {
                    if (parent_player == 1)
                        pred_input.push_back(1.0);
                    else
                        pred_input.push_back(-1.0);
                }

                if (ch == 'b')
                {
                    if (parent_player == 1)
                        pred_input.push_back(-1.0);
                    else
                        pred_input.push_back(1.0);
                }
                if (ch == 'A')
                {
                    if (parent_player == 1)
                        pred_input.push_back(2.0);
                    else
                        pred_input.push_back(-2.0);
                }

                if (ch == 'B')
                {
                    if (parent_player == 1)
                        pred_input.push_back(-2.0);
                    else
                        pred_input.push_back(2.0);
                }
            }
        }
    }

    for (auto t: pred_input)
        std::cerr << t << ' ';
    std::cerr << endl;

    rval = net.predict(pred_input)[0];
    return rval;
}

/* copy numbytes from src to destination
   before the copy happens the destSize bytes of the destination array is set to 0s
*/
void safeCopy(char *dest, char *src, int destSize, int numbytes)
{
    memset(dest, 0, destSize);
    memcpy(dest, src, numbytes);
}

/* Employ your favorite search to find the best move here.  */
/* This example code shows you how to call the FindLegalMoves function */
/* and the PerformMove function */
void FindBestMove(int player, char board[8][8], char *bestmove)
{
    try
    {
        net.load("testnet");
    }
    catch (exception e)
    {
        fprintf(stderr, "There is no testnet to use. MLPPlayer needs a testnet. exiting...");
        exit(-1);
    };

    int bestMoveIndex;
    float bestEval = -DBL_MAX;
    struct State state;

    setupBoardState(&state, player, board);

    // Here's an example loop walking through all the moves in the board state
    for (int i = 0; i < state.numLegalMoves; i++)
    {
        State nextState;
        memcpy(&nextState, &state, sizeof(State));
        performMove(&nextState, i);

        double eval = get_board_eval(&nextState);
        if (eval > bestEval)
        {
            bestEval = eval;
            bestMoveIndex = i;
        }
    }

    // For now, until you write your search routine, we will just set the best move
    // to be a random (legal) one, so that it plays a legal game of checkers.
    // You *will* want to replace this with a more intelligent move seleciton
    // bestMoveIndex = rand() % state.numLegalMoves;

    safeCopy(bestmove, state.movelist[bestMoveIndex], MaxMoveLength, MoveLength(state.movelist[bestMoveIndex]));
}
