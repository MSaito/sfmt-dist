/*
 * This file is copied from
 * http://people.sc.fsu.edu/~jburkardt/cpp_src/ziggurat/ziggurat.html.
 * Modified by Mutsuo Saito (saito@manieth.com)
 *
 * License: the GNU LGPL license.
 *
 * Authors: George Marsaglia, Wai Wan Tsang.
 */
#include <cmath>
#include <cfloat>
#include <inttypes.h>
#include <iostream>
#include <iomanip>

using namespace std;

double fe[256];
double fn[128];
uint32_t ke[256];
uint32_t kn[128];
double we[256];
double wn[128];

/*************************************************************************/
/*
  Purpose:

  R4_EXP_SETUP sets data needed by R4_EXP.

  Licensing:

  This code is distributed under the GNU LGPL license.

  Modified:

  04 October 2013

  Author:

  George Marsaglia, Wai Wan Tsang.
  Modifications by John Burkardt.

  Reference:

  Philip Leong, Guanglie Zhang, Dong-U Lee, Wayne Luk, John Villasenor,
  A comment on the implementation of the ziggurat method,
  Journal of Statistical Software,
  Volume 12, Number 7, February 2005.

  George Marsaglia, Wai Wan Tsang,
  The Ziggurat Method for Generating Random Variables,
  Journal of Statistical Software,
  Volume 5, Number 8, October 2000, seven pages.

  Parameters:

  Global, uint32_t KE[256], data needed by R4_EXP.

  Global, float FE[256], WE[256], data needed by R4_EXP.
*/
void r4_exp_setup ( )
{
    double de = 7.697117470131487;
    int i;
    const double m2 = 4294967296.0;
    double q;
    double te = 7.697117470131487;
    const double ve = 3.949659822581572e-03;

    q = ve / exp ( - de );

    ke[0] = ( uint32_t ) ( ( de / q ) * m2 );
    ke[1] = 0;

    we[0] =  ( q / m2 );
    we[255] =  ( de / m2 );

    fe[0] = 1.0;
    fe[255] =  ( exp ( - de ) );

    for ( i = 254; 1 <= i; i-- ) {
        de = - log ( ve / de + exp ( - de ) );
        ke[i+1] = ( uint32_t ) ( ( de / te ) * m2 );
        te = de;
        fe[i] =  ( exp ( - de ) );
        we[i] =  ( de / m2 );
    }
}

/*************************************************************************/
/*
  Purpose:

  R4_NOR_SETUP sets data needed by R4_NOR.

  Licensing:

  This code is distributed under the GNU LGPL license.

  Modified:

  20 May 2008

  Author:

  George Marsaglia, Wai Wan Tsang.
  Modifications by John Burkardt.

  Reference:

  Philip Leong, Guanglie Zhang, Dong-U Lee, Wayne Luk, John Villasenor,
  A comment on the implementation of the ziggurat method,
  Journal of Statistical Software,
  Volume 12, Number 7, February 2005.

  George Marsaglia, Wai Wan Tsang,
  The Ziggurat Method for Generating Random Variables,
  Journal of Statistical Software,
  Volume 5, Number 8, October 2000, seven pages.

  Parameters:

  Global, uint32_t KN[128], data needed by R4_NOR.

  Global, float FN[128], WN[128], data needed by R4_NOR.
*/

void r4_nor_setup ( )
{
    double dn = 3.442619855899;
    int i;
    const double m1 = 2147483648.0;
    double q;
    double tn = 3.442619855899;
    const double vn = 9.91256303526217e-03;
    /*
      Set up the tables for the normal random number generator.
    */
    q = vn / exp ( - 0.5 * dn * dn );

    kn[0] = ( uint32_t ) ( ( dn / q ) * m1 );
    kn[1] = 0;

    wn[0] =  ( q / m1 );
    wn[127] =  ( dn / m1 );

    fn[0] = 1.0;
    fn[127] =  exp ( - 0.5 * dn * dn );

    for ( i = 126; 1 <= i; i-- )
    {
        dn = sqrt ( - 2.0 * log ( vn / dn + exp ( - 0.5 * dn * dn ) ) );
        kn[i+1] = ( uint32_t ) ( ( dn / tn ) * m1 );
        tn = dn;
        fn[i] =  exp ( - 0.5 * dn * dn );
        wn[i] =  ( dn / m1 );
    }
}

void output_float(const char * str, double data[], int length)
{
    cout << endl;
    cout << str << " = {";
    for (int i = 0; i < length; i++) {
        if (i % 2 == 0) {
            cout << endl;
        }
        cout << scientific << setprecision(16) << data[i] << ", ";
    }
    cout << "};" << endl;
}

void output_int(const char * str, uint32_t data[], int length)
{
    cout << endl;
    cout << str << " = {";
    for (int i = 0; i < length; i++) {
        if (i % 4 == 0) {
            cout << endl;
        }
        cout << dec << data[i] << ", ";
    }
    cout << "};" << endl;
}

int main() {
    r4_nor_setup();
    r4_exp_setup();
    cout << "#include <inttypes.h>" << endl;
    cout << "#include \"ziggurat1.h\"" << endl;
    cout << "namespace Ziggurat {" << endl;
    output_float("const float fe[256]", fe, 256);
    output_int("const uint32_t ke[256]", ke, 256);
    output_float("const float we[256]", we, 256);
    output_float("const float fn[128]", fn, 128);
    output_int("const uint32_t kn[128]", kn, 128);
    output_float("const float wn[128]", wn, 128);
    cout << "}" << endl;
}
