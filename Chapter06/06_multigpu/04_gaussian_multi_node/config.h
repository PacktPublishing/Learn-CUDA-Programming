#ifndef CONFIG_H
    #define CONFIG_H

    #define INPUT_TYPE RANDOM
    #define PIVOTPACK
    #define ROWS 33000           // Number of rows in the system
    #define COLS 30000           // Number of columns in the system
    #define PERCENTAGE 50       // Density of coefficient matrix. Useful only with INPUT_TYPE set to RANDOM



    #define REFERENCE_SOLUTION "original-matrix"
    #define COMPUTED_SOLUTION  "computed-solution"

    // Chose one of the two
    // 32 consecutive matrix elements are packed together in an unsigned int
    // #define ELEMENT_TYPE_UINT
    // 128 consecutive matrix elements are packed together in an uint4
       #define ELEMENT_TYPE_UINT4
#endif
