#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int isPalindrome(int);

int main() {
    int max = -1; 
    //printf("986689 is palindrome? %d", isPalindrome(986689));
    //printf("642246 is palindrome? %d", isPalindrome(642246));
    for(int baseB = 999; baseB>700; baseB--) {
        for(int baseA = 999; baseA>700; baseA--) {
            int sum = baseA*baseB;
            if(isPalindrome(sum)&&sum>max) {
                max = sum;
            }
        }
    }
    printf("%d\n",max);
}


//Assumes the number is 6 digits long. Hacky.
int isPalindrome(int x) {
    char *digits = malloc(6);
    int i = 0;
    while(x) {
        int result = x % 10;
        //printf("%d\n", result);
        x/=10;
        digits[i++] = result;
    }
    if(digits[0]!=digits[5]) {
        return 0;
    } else if(digits[1]!=digits[4]) {
        return 0;
    } else if(digits[2]!=digits[3]) {
        return 0;
    }
    return 1;

}
