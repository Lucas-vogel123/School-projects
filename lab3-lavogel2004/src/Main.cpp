/**
 * Main.cpp
 * CSE 332S
 * Lab Num: 3
 * Author: Lucas Vogel
*/

#include <iostream>
#include "tictactoe.h"
using namespace std;

// Error message that returns enum for invalid argument
int usageMessage(const std::string &programName, const std::string &usageInfo) {
    std::cout << "Usage Message: " << programName << " " << usageInfo << std::endl;
    return invalidArguments;
}

int main(int argc, char *argv[]) {
    bool auto_player = false;
    if(argc > 3 || argc < 2 || argv[1] != string("TicTacToe")) {
        return usageMessage(argv[0], "invalid arguments: EX: ./lab_x TicTacToe");
    }
    if(argv[2] != nullptr) {
        if(argv[2] == string("auto_player")) {
            auto_player = true;
        }
        else {
            return usageMessage(argv[0], "invaid arguments: EX: ./lab_x TicTacToe auto_player");
        }
    }


    tictactoe game;
    return game.play(auto_player); // runs Tic Tac Toe Game
}


