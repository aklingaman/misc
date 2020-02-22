#include <stdio.h>
#include <stdlib.h>
    

int main() {
    int ans = 2;
    int fibLow = 1;
    int fibHigh = 2;
    while(1) {
        int swap = fibHigh;
        fibHigh = fibLow+fibHigh;
        fibLow = swap;
        if(fibHigh>4000000) {
            break;
        }
        if(fibHigh%2==0) {
            ans+=fibHigh;
        }
        
    }
    printf("%d\n",ans);
     



}
