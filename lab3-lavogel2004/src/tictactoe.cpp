//
// Lab 3 Lucas Vogel tictactoe.cpp
//

#include "tictactoe.h"
#include <sstream>

// Constructor for game - initializing board size
tictactoe::tictactoe() : board(board_size, vector<char>(board_size, ' ')), player('X'){
    for (int i = 0; i < board_size; i++) {
        board[0][i] = board[i][0] = ' ';
        board[4][i] = board[i][4] = ' ';
    }
}
ostream& operator<<(ostream& o, const tictactoe& game) {
    for (int i = tictactoe::board_size - 1; i >= 0; --i) {
        o << i << " ";
        for (int j = 0; j < tictactoe::board_size; ++j) {
            o << game.board[i][j] << " ";
        }
        o << endl;
    }
    o << "  ";
    for (int i = 0; i < tictactoe::board_size; ++i) {
        o << i << " ";
    }
    o << endl;
    return o;
}
//checks if board is in a state of winning
bool tictactoe::done() {
    bool result = false;
    for (int i = 1; i < board_size - 1; i++){
        if(board[i][1] != ' ' && board[i][1] == board[i][2] && board[i][2] == board[i][3]) {
            result = true;
        }
    }

    for(int j = 1; j < board_size - 1; j++) {
        if(board[1][j] != ' ' && board[1][j] == board[2][j] && board[2][j] == board[3][j]) {
            result = true;
        }
    }
    if(board[1][1] != ' ' && board[1][1] == board[2][2] && board[2][2] == board[3][3]) {
        result = true;
    }
    if(board[1][3] != ' ' && board[1][3] == board[2][2] && board[2][2] == board[3][1]) {
        result = true;
    }
    return result;
}
// Checks if board is full and no winner
bool tictactoe::draw() {
    bool full = true;
    for(int i = 1; i < board_size - 1; i++) {
        for(int j = 1; j < board_size - 1; j++) {
            if(board[i][j] == ' ') {
                full = false;
            }
        }
    }
    if(full) {
        if(done()) {
            return false;
        }
        else {
            return true;
        }
    }
    return false;
}
// Prompts the user to get user input for game
bool tictactoe::prompt(unsigned int& x, unsigned int& y) {
    // if stays false user asked to quit, else it should be true when the user produces a good input
    string userInput;
    while (true) {
        cout << "Enter coordinates x and y as 'x, y', or enter quit to leave: ";
        getline(cin, userInput);
        if(userInput == "quit") {
            return false;
        }
        stringstream ss(userInput);
        char c;
        if(ss >> x >> c >> y) {
            if(c == ',' && x < board_size - 1 && x >= 1 && y < board_size - 1 && y >= 1 && board[y][x] == ' ') {
                return true;
            }
            else {
                cout << "Invalid input. Try Again." << endl;
            }
        }
        else {
            cout << "invalid format. Please enter in 'x,y' format" << endl;
        }
    }

}

bool tictactoe::turn(bool autoPlayer) {
    // takes in auto_player bool to see which method of game is being played by user
    if(autoPlayer) {
        cout << "Player: " << player << "'s turn." << endl;
        unsigned int x, y;
        if(player == 'O') {
            pair<int, int> result = auto_player();
            x = result.first;
            y = result.second;
            board[y][x] = 'O';
            player = 'X';
            return true;
        }
        else {
            while(true) {
                if(!prompt(x, y)) {
                    return false;
                }
                board[y][x] = player;
                player = 'O';
                return true;
            }
        }
    }
    else {
        cout << "Player: " << player << "'s turn." << endl;
        unsigned int x, y;
        while (true) {
            if (!prompt(x, y)) {
                return false;
            }
            board[y][x] = player;
            player = (player == 'X') ? 'O' : 'X';
            return true;
        }
    }
}
// Minimax algorithm used for auto_player
int tictactoe::minimax(bool isMax) {
    if(done()) {
        return (player == 'O' ? -1 : 1);
    }
    if(draw()) {
        return drawed;
    }
    int best = isMax ? -1000 : 1000;
    for(int i = 1; i < board_size - 1; i++) {
        for(int j = 1; j < board_size - 1; j++) {
            if(board[i][j] == ' ') {
                board[i][j] = isMax ? 'O' : 'X';
                int score = minimax(!isMax);
                board[i][j] = ' ';

                if(isMax) {
                    best = max(best, score);
                }
                else {
                    best = min(best, score);
                }
            }
        }
    }
    return best;
}
// generates auto_players move based on minimax decision
pair<int, int> tictactoe::auto_player() {
    int best = -1000;
    pair<int, int> move = {-1, -1};

    for(int i = 1; i < board_size - 1; i++) {
        for(int j = 1; j < board_size - 1; j++) {
            if(board[i][j] == ' ') {
                board[i][j] = 'O';
                int score = minimax(false);
                board[i][j] = ' ';

                if(score > best) {
                    best = score;
                    move = {j, i};
                }
            }
        }
    }
    return move;
}

int tictactoe::play(bool autoPlayer) {
    while(true) {
        cout << *this << endl;
        if(!turn(autoPlayer)) {
            cout << "Game has ended" << endl;
            return quit;
        }
        if(done()) {
            cout << *this << endl;
            cout << "Player " << (player == 'X' ? "O" : "X") << " has won!" << endl;
            return playerWon;
        }
        if(draw()) {
            cout << *this << endl;
            cout << "game has ended in a draw!" << endl;
            return drawed;
        }
    }
}