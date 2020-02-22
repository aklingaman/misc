#include <stdio.h>
#include <stdlib.h>
#include <math.h>


long compute (long num);

int main() {
    int ans = -1; 
    long num = 600851475143;
    long root = (long) sqrt(num); 
    root+=(root/100); //Add to its precision, we dont need to check some of these values but its not going to hurt runtime much
    int longestFactor = 1;
    int check = 2;
    while(check<root) {
        if(num%check==0) {
            longestFactor = check;
            num/=check;
        } 
        check++;
    }
   printf("%ld\n", longestFactor);
}
