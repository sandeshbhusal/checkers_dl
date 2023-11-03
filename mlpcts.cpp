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
std::chrono::time_point<std::chrono::steady_clock> start_time;
int expansions = 0;

bool stop() {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start_time).count() >= SecPerMove * 1000 - 100;
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
class Node
{
public:
    struct State *state;
    vector<Node *> branches;
    Node *parent;
    double won = 0;
    double played = 0;
    int depth = 0;

public:
    bool gameOver()
    {
        return this->state->numLegalMoves == 0;
    }

public:
    bool isLeaf()
    {
        return this->branches[0] == nullptr;
    }

public:
    Node(struct State *state, Node *parent)
    {
        this->state = state;
        for (int i = 0; i < state->numLegalMoves; i++)
            this->branches.push_back(nullptr);
        this->parent = parent;
        if (parent != nullptr)
        {
            this->depth = this->parent->depth + 1;
        }
    }

public:
    int playout(struct State* state, int maxMoves) {
        int myBestMoveIndex = 0;
        if (state->numLegalMoves == 0){
            return -1;
        }
        if (maxMoves == 0)
            return 0;

        myBestMoveIndex = rand() % state->numLegalMoves;

        struct State nextstate;
        memcpy(&nextstate, state, sizeof(struct State));
        performMove(&nextstate, myBestMoveIndex);
        return -playout(&nextstate, maxMoves - 1);
    }

public:
    void expand()
    {
        if (stop()) return;
        expansions += 1;
        Node *current = this;
        for (int i = 0; i < current->state->numLegalMoves; i++)
        {
            if (stop()) return;

            struct State *next = (struct State *)(malloc(sizeof(struct State)));
            memcpy(next, current->state, sizeof(struct State));
            performMove(next, i);

            Node *newNode = new Node(next, current);

            // Predict the neural net output here.
            double temp = this->playout(this->state, 50);
            // End of prediction.

            if (temp > 0)
                newNode->won = 1.0;
            else
                newNode->won = 0;

            newNode->played = 1;
            current->branches[i] = newNode;
        }

        // Iteratively backprop the node results.
        for (int i = 0; i < current->state->numLegalMoves; i++)
        {
            if (stop()) return;
            Node *child = current->branches[i];
            double win = child->won;

            while (child->parent != nullptr)
            {
                win = 1 - win;
                child->parent->played += 1;
                child->parent->won += win;
                child = child->parent;
            }
        }
    }

public:
    double utility()
    {
        vec_t inputs_to_net = generate_input_from_board(this->state->player, this->state);
        float net_pred = net.predict(inputs_to_net)[0];
        return net_pred + 2.5 * sqrt(log(this->parent->played) / this->played);
    }

public:
    void select()
    {
        if (stop()) return;

        Node *current = this;
        if (current->gameOver())
        {
            double played = 1;
            double won = 0;
            while (current != nullptr)
            {
                current->won += won;
                current->played += played;
                won = 1.0 - won;
                current = current->parent;
            }

            return;
        }

        if (current->isLeaf())
        {
            current->expand(); 
            return;
        }

        double bestutility = current->branches[0]->utility();
        Node *selected = current->branches[0];
        for (int i = 0; i < current->branches.size(); i++)
        {
            double tempUtility = current->branches[i]->utility();
            if (tempUtility > bestutility)
            {
                bestutility = tempUtility;
                selected = current->branches[i];
            }
        }

        selected->select();
    }
};


/* copy numbytes from src to destination
   before the copy happens the destSize bytes of the destination array is set to 0s
*/
void
safeCopy(char *dest, char *src, int destSize, int numbytes)
{
    memset(dest, 0, destSize);
    memcpy(dest, src, numbytes);
}

void FindBestMove(int player, char board[8][8], char *bestmove)
{
    // Initialize the net.
    try {
        net.load("testnet");
    } catch(exception e) {
        cerr << "Could not load neural net." << endl;
    }

    // Initialize the timer.
    start_time = std::chrono::steady_clock::now();

    int bestMoveIndex;
    struct State state;

    setupBoardState(&state, player, board);

    // On win or loss, we don't need to do anything. The selflearncheckers program will update the
    // neural network params accordingly.

    // Run a thread, and get the value from player.
    Node *rootnode = new Node(&state, nullptr);
    while (!stop())
    {
        rootnode->select();
    }

    int maxplayed = -1000;
    for (int i = 0; i < rootnode->branches.size(); i++)
    {
        int plays = rootnode->branches[i]->played;
        if (plays > maxplayed) {
            bestMoveIndex = i;
            maxplayed = plays;
        }
    }
    cerr << "Before returning a move, "<< expansions<< " nodes were expanded.";
    safeCopy(bestmove, state.movelist[bestMoveIndex],MaxMoveLength, MoveLength(state.movelist[bestMoveIndex]));
}
