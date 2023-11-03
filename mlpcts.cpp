#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <sys/types.h>
#include <sys/times.h>
#include <time.h>
#include <float.h>
#include <vector>
#include <random>
#include "myprog.h"
#include "playerHelper.c"
#include "defs.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace std;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;
using namespace tiny_dnn;

network<sequential> net;

// Given a state, flip the board in the state, but not the player.
// This way, we always have positive values on the front of the board,
// and can do convolutions (checkers is symmetric).
void flipBoard(struct State *state)
{
    for (int y = 0; y < 4; y++)
    {
        for (int x = 0; x < 8; x++)
        {
            if (x % 2 != y % 2)
            {
                int temp = state->board[y][x];
                state->board[y][x] = state->board[7 - y][7 - x];
                state->board[7 - y][7 - x] = temp;
            }
        }
    }
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

// Generate inputs for the neural net - output is a 32x1 list,
// flips the board if player is player 2.
vec_t generate_input_from_board(int player, struct State *state)
{
    vec_t rval;

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

    if (player == 2)
    {
        flipBoard(state);
    }

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



vector<State*> generate_children(struct State *node) {
    vector<State*> rval;
    for (int i = 0; i < node->numLegalMoves; i++){
        State *newstate = new State;
        memcpy(newstate, node, sizeof(struct State));
        performMove(newstate, i);
        rval.push_back(newstate);
    }
    return rval;
}

void FindBestMove(int player, char board[8][8], char *bestmove)
{
    // Initialize the net.
    try
    {
        net.load("testnet");
    }
    catch (exception e)
    {
        cerr << "Could not load neural net." << endl;
        exit(0);
    }

    int bestMoveIndex;
    struct State state;

    setupBoardState(&state, player, board);
    vector<struct State*> children = generate_children(&state);

    if (children.size() == 0)
        cerr << "lost the game!" << endl;

    double bestEval = -10.0; // This cannot be reached by the NN - outputs should be in range of -1 to 1.
    
    vector<float> action_values;
    float total = 0.0;
    for (int i = 0; i < children.size(); i++) {
        double eval = net.predict(generate_input_from_board(player, children[i]))[0];
        // action_values.push_back(eval);
        if (eval > bestEval) {
            bestEval = eval;
            bestMoveIndex = i;
        }
    }

    // for (auto k: action_values)
    //     total += k;

    // for (int i = 0; i < action_values.size(); i++)
    //     action_values[i] /= total;

    // std::sort(action_values.begin(), action_values.end());

    // // roll a dice.
    // std::random_device rng;
    // std::mt19937 gen(rng());
    // std::uniform_real_distribution<> dis(0.0, 1.0);
    // double random_number = dis(gen);
    // cerr << "Dice roll: " << random_number << endl;
    // for (auto k : action_values)
    //     cerr << k << " ";
    // cerr << endl;

    // int i = 0;
    // while (i < action_values.size() && action_values[i] < random_number)
    //     i++;

    // bestMoveIndex = i - 1;
    // if (bestMoveIndex == -1){
    //     cerr << "Only can take piece" << endl;
    //     bestMoveIndex = 0;
    // }
    // cerr << "selected " << bestMoveIndex << endl;
    safeCopy(bestmove, state.movelist[bestMoveIndex], MaxMoveLength, MoveLength(state.movelist[bestMoveIndex]));
}
