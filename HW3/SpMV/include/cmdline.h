
#pragma once
#include <string.h>

char * get_arg(int argc, char ** argv, const char * key)
{
    char * val = NULL;

    size_t keylen = strlen(key);

    for(int i = 1; i < argc; i++){
        char * token = argv[i];
        
        if(strncmp(token, "--", 2) != 0)
            continue;
        token += 2;

        if(strncmp(token, key, keylen) != 0)
            continue;
        token += keylen;

        val = argv[i];
    }
    
    return val;
}

char * get_argval(int argc, char ** argv, const char * key)
{
    char * val = NULL;

    size_t keylen = strlen(key);

    for(int i = 1; i < argc; i++){
        char * token = argv[i];
        
        if(strncmp(token, "--", 2) != 0)
            continue;
        token += 2;

        if(strncmp(token, key, keylen) != 0)
            continue;
        token += keylen;

        if(strncmp(token, "=", 1) != 0)
            continue;
        token += 1;

        val = token;
    }
    
    return val;
}

