#ifndef COMPUTER_H
#define COMPUTER_H

// #define Empty 0x00
// #define Piece 0x20
// #define King 0x60
// #define Red 0x00
// #define White 0x80

// #define number(x) ((x)&0x1f)
// #define empty(x) ((((x)>>5)&0x03)==0?1:0)
// #define piece(x) ((((x)>>5)&0x03)==1?1:0)
// #define king(x) ((((x)>>5)&0x03)==3?1:0)

// // Be careful with this - it returns a 1 or 2, i.e. player number, where 1 corresponds to red and 2 to white
// #define color(x) ((((x)>>7)&1)+1)

// #define Clear 0x1f

// #define MaxMoveLength 12

enum {
    Empty = 0x00,
    Piece = 0x20,
    King = 0x60,
    Red = 0x00,
    White = 0x80,
    Clear = 0x1f,
    MaxMoveLength = 12
};

static int number(int x); // function declaration
static int empty(int x); // function declaration
static int piece(int x); // function declaration
static int king(int x); // function declaration
static int color(int x); // function declaration

int number(int x) {
    return (x & 0x1f);
}

int empty(int x) {
    return ((((x) >> 5) & 0x03) == 0) ? 1 : 0;
}

int piece(int x) {
    return ((((x) >> 5) & 0x03) == 1) ? 1 : 0;
}

int king(int x) {
    return ((((x) >> 5) & 0x03) == 3) ? 1 : 0;
}

int color(int x) {
    return ((((x) >> 7) & 1) + 1);
}

typedef struct State {
    int player;
    char board[8][8];
    char movelist[48][MaxMoveLength]; /* The following comments were added by Tim Andersen
                  Here Scott has set the maximum number of possible legal moves to be 48.
                                  Be forewarned that this number might not be correct, even though I'm
                                  pretty sure it is.  
                  The second array subscript is (supposed to be) the maximum number of 
                  squares that a piece could visit in a single move.  This number is arrived
                  at be recognizing that an opponent will have at most 12 pieces on the 
                  board, and so you can at most jump twelve times.  However, the number
                  really ought to be 13 since we also have to record the starting position
                  as part of the move sequence.  I didn't code this, and I highly doubt 
                  that the natural course of a game would lead to a person jumping all twelve
                  of an opponents checkers in a single move, so I'm not going to change this. 
                  I'll leave it to the adventuresome to try and devise a way to legally generate
                  a board position that would allow such an event.  
                  Each move is represented by a sequence of numbers, the first number being 
                  the starting position of the piece to be moved, and the rest being the squares 
                  that the piece will visit (in order) during the course of a single move.  The 
                  last number in this sequence is the final position of the piece being moved.  */
    int numLegalMoves;
} State;

void CopyState(char *dest, char src);
void ResetBoard(void);
void AddMove(char move[12]);
void FindKingMoves(char board[8][8], int x, int y);
void FindMoves(int player, char board[8][8], int x, int y);
void AddJump(char move[12]);
int FindKingJump(int player, char board[8][8], char move[12], int len, int x, int y);
int FindJump(int player, char board[8][8], char move[12], int len, int x, int y);
int FindLegalMoves(struct State *state);
int EVAL(char Board[8][8]);
int MaxVal(struct State *state, int alpha, int beta, int depth);
int MinVal(struct State *state, int alpha, int beta, int depth);
void FindBestMove(int player, char board[8][8], char bestmove[12]);
void NumberToXY(char num, int *x, int *y);
int MoveLength(char move[12]);
int TextToMove(char *mtext, char move[12]);
void MoveToText(char move[12], char *mtext);
void PerformMove(char board[8][8], char move[12], int mlen);
void performMove(State *state, int moveIndex);
void setupBoardState(struct State *state, int player, char board[8][8]);

#endif
