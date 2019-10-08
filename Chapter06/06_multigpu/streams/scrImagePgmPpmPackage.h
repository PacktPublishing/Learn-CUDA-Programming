#ifndef _Xiang_Gao_PGM_PPM_Header_
#define _Xiang_Gao_PGM_PPM_Header_

#include <stdio.h>


 int scr_read_pgm( char* name, unsigned char* image, int irows, int icols );
 void scr_write_pgm( char* name, unsigned char* image, int rows, int cols, char* comment );
 int scr_read_ppm( char* name, unsigned char* image, int irows, int icols );
 void scr_write_ppm( char* name, unsigned char* image, int rows, int cols, char* comment );
 void get_PgmPpmParams(char * , int *, int *);
 void getout_comment(FILE * );
#endif
