#include <stdio.h>
#include <stdlib.h>
#include <math.h>


long check(long);

int main() {
    long min = 2520;
    while(1) {
        min+=20; //2520 is divisble by 20, so we cant have that happen until 20 later, so only check every 20. 
        if(check(min)) { printf("%ld\n", min); break;}
    }
    
}


long check(long x) {
    for(int i = 3; i<20; i++) {
        if(i==4||i==5||i==10) { continue; } //A number div by 20 is also div by its factors. 
        if(x%i!=0) { return 0; }
    }
    return 1;
}
