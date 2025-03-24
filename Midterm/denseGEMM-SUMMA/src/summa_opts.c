#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include "summa_opts.h"

void print_usage(const char* prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  -m, --rows INT      Number of rows in matrix A\n");
    printf("  -n, --cols INT      Number of columns in matrix B\n");
    printf("  -k, --inner INT     Inner dimension (A cols/B rows)\n");
    printf("  -b, --block INT     Block size for tiled operations\n");
    printf("  -s, --stationary    Algorithm variant ('a' or 'b')\n");
    printf("  -v, --verbose       Print detailed output\n");
    printf("  -p, --perf         Print performance metrics\n");
    printf("  -h, --help         Print this help\n");
}

SummaOpts parse_args(int argc, char *argv[]) {
    SummaOpts opts = {
        .m = 0,
        .n = 0,
        .k = 0,
        .block_size = 32,
        .stationary = 'a',
        .verbose = 0,
        .metrics = 0,
    };

    struct option long_options[] = {    // option is a default structure in C. Each row below is one option
        {"rows",      required_argument, 0, 'm'},   // 3rd parameter is a flag, if non-NULL, it points to a variable to be set. If NULL, ignored.
        {"cols",      required_argument, 0, 'n'},
        {"inner",     required_argument, 0, 'k'},
        {"block",     required_argument, 0, 'b'},
        {"stationary",required_argument, 0, 's'},
        {"verbose",   no_argument,       0, 'v'},
        {"perf",      no_argument,       0, 'p'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    // getopt_long is a function provided by GNU C that allows parsing command line options
    while ((opt = getopt_long(argc, argv, "m:n:k:b:s:vph", long_options, NULL)) != -1) {
        switch (opt) {
            case 'm': opts.m = atoi(optarg); break;
            case 'n': opts.n = atoi(optarg); break;
            case 'k': opts.k = atoi(optarg); break;
            case 'b': opts.block_size = atoi(optarg); break;
            case 's': opts.stationary = optarg[0]; break;
            case 'v': opts.verbose = 1; break;
            case 'p': opts.metrics = 1; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: print_usage(argv[0]); exit(1);
        }
    }

    // Validate parameters
    if (opts.m <= 0 || opts.n <= 0 || opts.k <= 0) {
        printf("Error: Matrix dimensions must be positive\n");
        print_usage(argv[0]);
        exit(1);
    }

    if (opts.block_size <= 0) {
        printf("Error: Block size must be positive\n");
        print_usage(argv[0]);
        exit(1);
    }

    if (opts.stationary != 'a' && opts.stationary != 'c') {
        printf("Error: Stationary parameter must be 'a' or 'c'\n");
        print_usage(argv[0]);
        exit(1);
    }

    return opts;
}