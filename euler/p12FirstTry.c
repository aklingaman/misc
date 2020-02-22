#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int facNum(long num);



int main() { 
    long num = 1;
    int delta = 2;
    int max = 2;
    //What Ive learned. No sense in not trying the brute force solution if you dont know if its going to take heat death of the universe time.
    //What ive also learned:  brute force doesnt work...... at least not without a few shortcuts. 
    while(1) {
       // printf("check: %ld\n",i);
        int test = facNum(num);
        if(test<501) {
            if(test>max) {
                max = test;
                printf("new max: %ld with %d factors\n", num,max);
            }
            num+=delta;
            delta++;
        } else {
            printf("ans: %ld", num);
            exit(0);
        }   
    }
      
}


//Shortcut problems: If we hardcode the factors of numbers we wont know what the factors are, so any new numbers have to be analyzed all the way through anyways. 
//That doesnt mean we cannot however realize that the number will have few factors, and just not bother with calculating them out. 
//Question: what feature of a number lets us eliminate it from possibly having 500 factors.    
//idea: track operations and find the cheapest way to get to the ans.
//idea: find a way to use 2 or fewer multiplications to replace the modulu 
int facNum (long num) {
    FILE *primes = fopen("primeNumbers.txt","r");
    int ret = 2; //1 and num itself are implicitly counted.
    int prime;
    fscanf(primes, "%d", &prime);
    //printf("facNum: %ld\n",num);
    while(num!=1) {
        //printf("\t checking prime: %d\n",prime);
        if(num==prime) {
            fclose(primes);
            return ret;
        }
        if(num%prime==0) {
            ret++;
            num/=prime;
        } else {
            fscanf(primes,"%d", &prime);
        }
    }
    fclose(primes);
    return ret;
    
}


