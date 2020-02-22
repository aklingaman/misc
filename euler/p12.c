#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    printf("This program takes like 10 mins or something idk\n");
    int max = 4;
    int num = 6;
    int delta = 4;
    while(1) {
        num+=(delta++);
        int sum = 2;
        
        if(num&0x01==0) { //Bit shift lets us shortcut all evenness out of the #
            num>>1;
            sum++;
        }

        for(int i = 2; i<num/2; i++) {
            if(num%i==0) {
                sum++;
            }
        }
        if(sum>max) {
            max = sum;
            printf("new max: %d\n", max);
            if(max>500) {
                printf("ans: %d\n",num);
            }
        }
    }

} 
