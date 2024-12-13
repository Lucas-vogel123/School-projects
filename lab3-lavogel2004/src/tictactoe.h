// Lucas Vogel Lab 3 tictactoe.h

#ifndef TICTACTOE_H
#define TICTACTOE_H

#include <vector>
#include <iostream>

using namespace std;

// Enum return values
enum returnValues {
    playerWon = 0,
    quit = 1,
    drawed = 2,
    invalidArguments = 3
};
// Class for tic tac toe game
class tictactoe {
private:
    static const int board_size = 5;
    vector<vector<char>> board;
    char player;
public:
    tictactoe();
    friend ostream& operator<<(ostream& os, const tictactoe& game);
    bool done();
    bool draw();
    bool prompt(unsigned int& x, unsigned int& y);
    bool turn(bool autoPlayer);
    int minimax(bool isMax);
    pair<int, int> auto_player();
    int play(bool autoplayer);
};


#endif //TICTACTOE_H
