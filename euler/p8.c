#include <stdio.h>
#include <stdlib.h>
#include <math.h>


long compute (long num); //If func doesnt exist this doesnt ruin anything

int main() {
    //File reading.
    FILE *open = fopen("p8num.txt","r");
    char num[1500];
    fgets(num,1000,open); //For some reason misses the 0 at end? No idea
    //printf("%s\n",num);
    //File reading 
    for(int i = 0; i<1000; i++) {
        num[i]-=48; //char -> int
    }
 
    long largestProduct = 0;
    int firstDigit = 0;
    //Idea: we can conclude changes to the product by looking at what we shift on and off, so we only need to compute the products from the crests!
    //Shift by one, review change, update if needed.
    for(int i = 0; i<987; i++) {
        for(int k = i; k<i+13; k++) {
            printf("%d",num[k]);
        }
        printf("-");
        if(num[i+12]==0) {
            i+=12; 
            printf("\n");
            continue;
        }
        if(num[i+13]>=num[i]) { //If the next number to add is larger than the one we would remove, the product will go up.
            printf("going up!\n");
            continue;
        } else {
            //Were going to go down on the next step, so we need to compute here to compare. 
            long product = 1;
            for(int j = i; j<i+13; j++) { 
                product *= (num[j]);
            }
            printf("\t%ld",product);
            if(product>largestProduct) { 
                largestProduct = product; 
                firstDigit = i; 
            }
        }
        printf("\n");
    }
    printf("largest: %ld\ndigits: ", largestProduct);    
    for(int i = 0; i<13; i++) {
        printf("%d ", num[i]);
    }   
    printf("\n");
}
