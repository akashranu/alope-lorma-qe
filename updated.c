#include <stdio.h>
#include "pico/stdlib.h"
#include <time.h>
#include <string.h>
#include "Includes/seven_seg_led.h"
//#include "includes/seven_segment.h"
// the example code is to display a message when a button is pressed

#define BUTTON_PIN 16   

void processMorse();
char morseCode[4];
char aLetter[2] = {'.-'};
char bLetter[4] = {'-...'};
int main() {

    stdio_init_all();
    welcome();
    gpio_init(BUTTON_PIN);
    gpio_set_dir(BUTTON_PIN, GPIO_IN);
    gpio_pull_down(BUTTON_PIN); //pull down the button pin towards ground
    clock_t start_t, end_t, after_t;
    double total_t;
    int count = 0;
    //bool action_triggered = false;

    while (true) {
        // pressed will be true if the button is currently being pressed
        bool pressed = gpio_get(BUTTON_PIN);
        bool held;
        if (pressed){
            seven_segment_show(0);
            sleep_ms(300);
            seven_segment_off();
            start_t = clock();
            held = true;
            while(held){
                held = gpio_get(BUTTON_PIN);
               
            }
            after_t = clock();
        end_t = clock();
        total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC * 1000;
        //printf("%lf \n",total_t);
       
        if (total_t < 1000.000000 && after_t < 700.000000 ){
            printf("dot \n");
            processMorse('.');
            strcat(morseCode, ".");
        } else if (total_t >= 1000.000000 && total_t <= 3000.000000 && after_t < 700.000000) {
            printf("dash \n");
            processMorse('-');
            strcat(morseCode, "-");
        } else if (after_t > 700.000000){
        processMorse('c');
        clearArray();
        seven_segment_show(1);
        sleep_ms(300);
        seven_segment_off();
        /*if (total_t < 250.000000){
            printf("dot \n");
            processMorse('.');
        } else if (total_t >= 300.000000 && total_t < 700.000000) {
            printf("dash \n");
            processMorse('-');
        } else
        seven_segment_show(1);
        sleep_ms(300);
        seven_segment_off();
*/
        } else if (total_t >= 700)

        sleep_ms(20);
    }
   
}
    void welcome(){
        printf("%s\n", "WELCOME!");
        seven_segment_init();
        seven_segment_show(0);
        sleep_ms(2000);
        seven_segment_off();
    }

    void clearArray(){
        for (int i = 0; i < arraySize; ++i) {
            myArray[i] = NULL;
            }
    }
    int compareArrays(char a[], char b[]) {
        bool output = true;
        size_t n1 = strlen(a);
        size_t n2 = strlen(b);

        if ( n1 != n2){
            output = false;
        } else {
            for (int i =0; i< n1;i++){
                if(a[i] != b[i]){
                    output = false;
                }
            }
        }
        ".-  "
        morseCode[0] == "."
        ".","-"
        return output;

}

void processMorse()) {
    static char morseSequence[10];
    static int sequenceIndex = 0;
    if (morse == 'c'){

    // Add the dot or dash to the sequence
   
    if(compareArrays(morseCode,aLetter) == true){
        printf("A");
    }else if(compareArrays(morseCode,bLetter)){
        printf("B");
    }
    }
    }