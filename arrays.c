#include <stdio.h>
#include "pico/stdlib.h"
#include <time.h>
#include <string.h>
#include "Includes/seven_seg_led.h"
//#include "includes/seven_segment.h"
// the example code is to display a message when a button is pressed

#define BUTTON_PIN 16 
//char morses[5] = {};
   int count = 0;
      bool clean_morse = false;
char morse_code[26][5] = {".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--", "-.", ""};

char alphabet[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
int main() {
    
    stdio_init_all();
    welcome();
    gpio_init(BUTTON_PIN);
    gpio_set_dir(BUTTON_PIN, GPIO_IN);
    gpio_pull_down(BUTTON_PIN); //pull down the button pin towards ground
    clock_t start_t, end_t;
    start_t = clock();
    end_t = clock();
    double total_t;
    //bool action_triggered = false;
    bool start_timer = true;
    bool stop_timer = false;
   
     while (true) {
        // pressed will be true if the button is currently being pressed
        bool pressed = gpio_get(BUTTON_PIN);
       
        if (pressed){
            if(start_timer){
                start_timer = false;
                start_t = clock();
                seven_segment_show(26);
                stop_timer = true;
                
                if(clean_morse){
                    clearMorseArray();
                    clean_morse = false;
                }
            }
        }
        else{
            if(stop_timer){
                stop_timer = false;
                printf("unpressed");
                end_t = clock();
                total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC * 1000;
                processTime(total_t);
                seven_segment_off();
                start_timer = true;
            } else{
               clock_t current_time = clock();
               double unpressed_t = (double)(current_time - end_t) / CLOCKS_PER_SEC * 1000;
               processUnpressedTime(unpressed_t);
            }

        }
        //sleep_ms(20);
    }
}
void welcome(){
    printf("%s\n", "WELCOME!");
    seven_segment_init();
    seven_segment_show(26);
    sleep_ms(2000);
    seven_segment_off();
}
void processTime(double total_t){
    if (total_t < 250.000000){
       
        processMorse('.');
    } else if(total_t <= 700.000000){
       
        processMorse('-');
    } else{
        seven_segement_show(27);
    }
}
void processMorse(char morse){
    if(count < 5){
        morses[count] = morse;
        count++;
    }
    else{
        printf("error: Morse size cannot be more than 5");
        clearMorseArray();
    }
}
void clearMorseArray(){
    int i = 0;
    for(i = 0; i < 5; i++){
        morses[i] = '\0';
    }
    count = 0;
}

void processLetters(char mCode[5]){
    bool code_match = false;
    int j = 0;
    for( j = 0; j < 26; j++){
        if(strcmp(mCode, morse_code[j]) == 0){
            printf("\n" alphabet[j]);
            seven_segement_show(values[j]);
            code_match =true;
        } 
    }
    if(code_match == false){
        printf("morse code is invalid: ");
        printMorse(mCode);
    }
}
void processUnpressedTime(double unpressed_t){
     if(unpressed_t > 700.000000){
        clean_morse = true;
        printMorse(morses);
        processLetters(morses);
    }
}
int getMorseLength(){
    int count = 0;
    for(count = 0; count < 5; count++){
        if(morses[count] == '\0'){
            break;
        }
    }
    return count;
}
void printMorse(char mCode[5]){
    int count = 0;
    for(count = 0; count < 5; count++){
        if(mCode[count] != '\0'){
            printf("%c",mCode[count]);
        }
    }
}