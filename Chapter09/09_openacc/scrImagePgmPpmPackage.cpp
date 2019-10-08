#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>


#include "scrImagePgmPpmPackage.h"

void get_string_nocomments( FILE* fdin, char* s )
{
	int i,done;
	
	done=0;
	while( !done )
	{
		fscanf( fdin, "%s", s );
		for( i=0; !done; ++i )
		{
			if( (s)[i] == '#' )
			{
				fgets( s, 256, fdin );
				break;
			}
			if( ! isspace(s[i]) )
				done = 1; 
		}
	}
}

int scr_read_pgm( char* name, unsigned char* image, int irows, int icols )
{
	int i,j, range, rows, cols;
	unsigned char* tpt;
	char*	ins;
	char	imgtype[4], whitespace;
	FILE*	fdin;

	ins = (char*)malloc(257);
	if( (fdin=fopen(name,"rb")) == NULL )
	{
		return 0;
	}

	fscanf( fdin, "%s", imgtype );

	get_string_nocomments( fdin, ins );
	sscanf( ins,"%d", &cols );

	get_string_nocomments( fdin,ins );
	sscanf( ins, "%d", &rows );

	get_string_nocomments( fdin,ins );
	sscanf( ins, "%d", &range );

	if( (rows<=0) || (cols<=0) )
		fprintf( stderr, ": negative rows or columns. Check!! %s", name );
	if( (irows!=rows) || (cols!=icols) )
	{
		fprintf( stderr, ": rows or columns don't match for file %s\n", name );
		fprintf( stderr, ": read %d %d, expected %d %d \n", rows,cols,irows,icols );
	}
	if( (strcmp(imgtype,"P5")==0) || (strcmp(imgtype,"p5")==0) )
	{
		fscanf( fdin, "%c", &whitespace );
		/* read binary form */
		tpt = image;
		for( i=0; i<rows; i++ )
		{
			for( j=0; j<cols; j++,tpt++ )
			{
				if( EOF == fscanf(fdin,"%c",tpt) ) 
					fprintf( stderr, "WARNING!! .not enought bytes for %s, on r%d c%d\n", name,i,j );
			}
		}
	}
	else if( (strcmp(imgtype,"P2")==0) || (strcmp(imgtype,"p2")==0) )
	{
		/* its ascii form */
		tpt = image;
		for( i=0; i<rows; i++ )
		{
			for( j=0; j<cols; j++,tpt++ )
			{
				if( EOF == fscanf(fdin,"%d",tpt) ) 
					fprintf( stderr, "WARNING!! .not enought bytes for %s, on r%d c%d\n", name,i,j );
			}
		}
	}
	else
	{
		/* not p2 or p5 */
		fprintf( stderr, "ERROR, called read pgm_byte but %s not a P2 or P5 image\n", name );
	}
	free( ins );
	fclose( fdin );
	return 1;
}

void scr_write_pgm( char* name, unsigned char* image, int rows, int cols, char* comment )
{
	int i,j;
	unsigned char* tpt;
	char fname[256];
	FILE* fdout;

	if( strlen(name) < 1 )
	{
		fprintf( stderr, "Image write called withoutd filename\n" );
		fprintf( stderr, "Please input filename->" );
		scanf( "%s", fname );
	}
	else
		strcpy( fname, name );
	if( (fdout=fopen(fname,"wb")) == NULL )
	{
		fprintf( stderr, "cannot open file >>%s<< for output\n", fname );
		fprintf( stderr, "continuing without write\n" );
	}
	if( comment )
		fprintf( fdout, "P5\n#%s \n%d %d\n255\n", comment, cols, rows );
	else
		fprintf( fdout, "P5\n%d %d\n255\n", cols, rows );
	tpt = image;
	for( i=0; i<rows; i++ )
	{
		for( j=0; j<cols; j++ )
		{
			fprintf( fdout, "%c", (unsigned char)*tpt++ );
		}
	}
	fprintf( fdout, "\n" );
	fclose( fdout );
}

int scr_read_ppm( char* name, unsigned char* image, int irows, int icols )
{
	int i,j, range, rows, cols;
	unsigned char* tpt;
	char*	ins;
	char	imgtype[4], whitespace;
	FILE*	fdin;

	ins = (char*)malloc(257);
	if( (fdin=fopen(name,"rb")) == NULL )
	{
		return 0;
	}

	fscanf( fdin, "%s", imgtype );
	get_string_nocomments( fdin, ins );
	sscanf( ins,"%d", &cols );

	get_string_nocomments( fdin,ins );
	sscanf( ins, "%d", &rows );

	get_string_nocomments( fdin,ins );
	sscanf( ins, "%d", &range );

	if( (rows<=0) || (cols<=0) )
		fprintf( stderr, ": negative rows or columns. Check!! %s", name );
	if( (irows!=rows) || (cols!=icols) )
	{
		fprintf( stderr, ": rows or columns don't match for file %s\n", name );
		fprintf( stderr, ": read %d %d, expected %d %d \n", rows,cols,irows,icols );
	}
	if( (strcmp(imgtype,"P6")==0) || (strcmp(imgtype,"p6")==0) )
	{
		fscanf( fdin, "%c", &whitespace );
		/* read binary form */
		tpt = image;
		for( i=0; i<rows; i++ )
		{
			for( j=0; j<cols; j++,tpt+=3 )
			{
				if( EOF == fscanf(fdin,"%c",tpt+2) ) 
					fprintf( stderr, "WARNING!! .not enought bytes for %s, on r%d c%d\n", name,i,j );
				if( EOF == fscanf(fdin,"%c",tpt+1) ) 
					fprintf( stderr, "WARNING!! .not enought bytes for %s, on r%d c%d\n", name,i,j );
				if( EOF == fscanf(fdin,"%c",tpt) ) 
					fprintf( stderr, "WARNING!! .not enought bytes for %s, on r%d c%d\n", name,i,j );
			}
		}
	}
	else if( (strcmp(imgtype,"P3")==0) || (strcmp(imgtype,"p3")==0) )
	{
		/* its ascii form */
		tpt = image;
		for( i=0; i<rows; i++ )
		{
			for( j=0; j<cols; j++,tpt+=3 )
			{
				if( EOF == fscanf(fdin,"%d",tpt+2) ) 
					fprintf( stderr, "WARNING!! .not enought bytes for %s, on r%d c%d\n", name,i,j );
				if( EOF == fscanf(fdin,"%d",tpt+1) ) 
					fprintf( stderr, "WARNING!! .not enought bytes for %s, on r%d c%d\n", name,i,j );
				if( EOF == fscanf(fdin,"%d",tpt) ) 
					fprintf( stderr, "WARNING!! .not enought bytes for %s, on r%d c%d\n", name,i,j );
			}
		}
	}
	else
	{
		/* not p3 or p6 */
		fprintf( stderr, "ERROR, called read pgm_byte but %s not a P3 or P6 image\n", name );
	}
	free( ins );
	fclose( fdin );
	return 1;
}

void scr_write_ppm( char* name, unsigned char* image, int rows, int cols, char* comment )
{
	int i,j;
	unsigned char* tpt;
	char fname[256];
	FILE* fdout;

	if( strlen(name) < 1 )
	{
		fprintf( stderr, "Image write called withoutd filename\n" );
		fprintf( stderr, "Please input filename->" );
		scanf( "%s", fname );
	}
	else
		strcpy( fname, name );
	if( (fdout=fopen(fname,"wb")) == NULL )
	{
		fprintf( stderr, "cannot open file >>%s<< for output\n", fname );
		fprintf( stderr, "continuing without write\n" );
	}
	if( comment )
		fprintf( fdout, "P6\n#%s \n%d %d\n255\n", comment, cols, rows );
	else
		fprintf( fdout, "P6\n%d %d\n255\n", cols, rows );
	// the order is r g b
	tpt = image;
	for( i=0; i<rows; ++i )
	{
		for( j=0; j<cols; ++j,tpt+=3 )
		{
			fprintf( fdout, "%c%c%c", (unsigned char)(*(tpt+2)), (unsigned char)(*(tpt+1)), (unsigned char)(*tpt) );
		}
	}
	fprintf( fdout, "\n" );
	fclose( fdout );
}


void get_PgmPpmParams(char* name, int *irows, int *icols )
{
	int range, rows, cols;
	char	imgtype[4];
	char*	ins;
	FILE*	fdin;
	ins = (char*)malloc(257);
	if( (fdin=fopen(name,"rb")) == NULL )
	{
		printf("File %s is not available - check\n",name);
		exit(1);
	}
	
	fscanf( fdin, "%s", imgtype );

	get_string_nocomments( fdin, ins );
	sscanf( ins,"%d", &cols );

	get_string_nocomments( fdin,ins );
	sscanf( ins, "%d", &rows );

	get_string_nocomments( fdin,ins );
	sscanf( ins, "%d", &range );

	if( (rows<=0) || (cols<=0) )
		fprintf( stderr, ": negative rows or columns. Check!! %s", name );

	*irows = rows;
	*icols = cols;
	fclose(fdin);
}

void getout_comment(FILE* fdin)
{
	char c;

	c = fgetc(fdin);
	do 
		c = fgetc(fdin);
	while (c != '\n');
}

