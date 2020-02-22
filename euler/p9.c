#include <stdio.h>
#include <stdlib.h>
#include <math.h>


long compute (long num); //If func doesnt exist this doesnt ruin anything

int main() {
    
    for(double a = 3; a<1000; a++) {
        for(double b = a+1; b<1000; b++) {
            double c = sqrt(a*a+b*b);
            
            if(fmod(c,1.0)) { //If c has a decimal portion, c % 1 != 0
                continue;
            }
            if((a+b+c==1000)) {
                printf("a: %f, b: %f\n",a,b);
            }
        }
    }


}
