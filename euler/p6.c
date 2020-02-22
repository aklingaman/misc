#include <stdio.h>
#include <stdlib.h>
#include <math.h>


long compute (long num); //If func doesnt exist this doesnt ruin anything

int main() {
    int sumSquare = 1;
    int squareSum = 1;
    
    for(int i = 2; i<101; i++) {
        sumSquare+=i;
        squareSum+=(i*i);
    }
    sumSquare = sumSquare*sumSquare;
    printf("%d\n", sumSquare-squareSum);
    
}
