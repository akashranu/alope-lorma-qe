#include <stdio.h>
#include "pico/stdlib.h"
#include <time.h>
#include <string.h>
#include "Includes/seven_seg_led.h"
//#include "includes/seven_segment.h"
// the example code is to display a message when a button is pressed

#define BUTTON_PIN 16 
char morses[5] = {};
   int count = 0;
      bool clean_morse = false;
//char morse_code[26][5] = {".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--"}

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
                seven_segment_show(0);
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
                printf("%d",total_t);
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
    seven_segment_show(0);
    sleep_ms(2000);
    seven_segment_off();
}
void processTime(double total_t){
    if (total_t < 250.000000){
       
        processMorse('.');
    } else if(total_t <= 700.000000){
       
        processMorse('-');
    } else{
        seven_segement_show(0b11111110);
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
void processUnpressedTime(double unpressed_t){
     if(unpressed_t > 700.000000){
        clean_morse = true;
        printMorse();
        if(morses[0] == '.' && getMorseLength() == 1){
            printf("E");
            seven_segment_show(6);
        }   
        if(morses[0] == '.' && morses[1] == '-' && morses[2] == '.' && getMorseLength() == 3){
             printf("R");
        }
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
void printMorse(){
    int count = 0;
    for(count = 0; count < 5; count++){
        if(morses[count] != '\0'){
            printf("%c",morses[count]);
        }
    }
}