
#pragma once

// A simple timer class
#include "time.h"
#include "rdtsc.h"
#include "../config.h"

typedef struct timer
{
    long long int start;
    long long int end;
} timer;

void timer_start(timer * t) 
{
    t->start = rdtsc();
}

float seconds_elapsed(timer * t)
{ 
    t->end = rdtsc();
    return (t->end - t->start)/FREQ_CPU;
}

float milliseconds_elapsed(timer * t)
{ 
    float elapsed_time;
    t->end = rdtsc();
    elapsed_time = 1000*(t->end - t->start)/FREQ_CPU;
    return elapsed_time;
}

