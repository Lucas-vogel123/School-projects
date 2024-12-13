# CSE 332 Lab Base
Repository for submitting lab work

Note: We will read this file as raw text rather than markdown.

Name: Lucas Vogel
Lab #: 3

Document your lab work here according to the lab specification
Errors and Logic:
- I did not run into any errors when building my code for the first time, but ran into logic errors in my game that I needed to fix:
- My axis were flipped when placing pieces. For example when I input "3,2" it would place the piece at "2,3". To fix this, i just reversed my x and y
inputs when placing the pieces down on the board.
- Another logic error I encountered was placing pieces outside of the board size, which I just needed to fix in my for loop by subtracting from the size.
Testing: 
- I ran the basic test cases first to see if it would run with a simple ./lab_x TicTacToe. This worked as accordingly and produced my Tic Tac Toe game.
- I then ran an empty string, a misspelled TicTacToe, and no TicTacToe key word. In all cases the code produced an error message with an example of how to run the program.
- I also tested my code with the user input while in the game with incorrect coordinates like "3, 2", "34" ",11", and the empty string " " and in all cases the program 
outputted either the error message or the correct coordinate on the board.
- I also tested the user input with coordinates outside of the board size like "5,5" or "4,3" and in all instances it gave an error message.
- In all cases of invalid inputs and incorrect number of command line arguments, the correct error message was printed and my program outputted a non 0 number indicating a failure. This is the correct behavior 
for the program.
Test Case Example: ./lab_x TicTacToe Output: Correct game output
Test Case Example: ./lab_x helloWorld Output: "Usage Message: ./lab_x invalid arguments: EX: ./lab_x TicTacToe"
Test Case Example: "4, 4" output: "Invalid input. Try Again."
Test Case Example: " " output: "invalid format. Please enter in 'x,y' format"
In each case it returned a none 0 value in the following enum:
  playerWon = 0,
  quit = 1,
  drawed = 2,
  invalidArguments = 3

Extra Credit:
I Designed my solution by using the minimax function which is a decision-making algorithm that is helpful to use in two player games.
I used resources like stackoverflow to help with my algorithm creation. The algorithm essentially constructs a tree where each node 
represents a possible board configuration, with the root node being the present state of the game. The auto-player in this case is aiming to
maximize its score while minimizing the opponents score. This eventually finds the optimal board that the auto_player wants to get to and makes 
the move based on that. In my case, the auto_player works some what well, but prioritizes it's own winning over defending the other player and 
can be easily beat. 
