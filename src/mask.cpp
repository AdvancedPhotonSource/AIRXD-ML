#include <bits/stdc++.h>

double Median(std::vector<double> a, int n){

    if (n % 2 == 0){
      nth_element(a.begin(),
                  a.begin() + n / 2,
                  a.end());

      nth_element(a.begin(),
                  a.begin() + (n - 1) / 2,
                  a.end());
    
      return (double) (a[(n-1)/2] + a[n/2])/2.0;
    }

    else {
      nth_element(a.begin(),
                  a.begin() + n / 2,
                  a.end());

      return (double) a[n/2];
    }
}

extern "C"
void mask(int M, int N, int chan, double dtth, double eps, int* tam, double* ta, double* band, double* tths, double* omask) {
  /* omask should be int. However, when I changed it to int, the result is garbage.
   * Can this still be improved? The four FOR loop takes the most. 
   * delete function for the vector. Why doesn't it work?
   */

    int h, i;
    double median, mad, temp;
    double scale = 1.4826;
    std::vector<long> val;
    std::vector<double> sband;
    std::vector<double> vmad;
    
    for (h=0; h<chan; h++){
      for (i=0; i<(M*N); i++){
        if (tam[i] != 1){
          if ((ta[i] >= tths[h]) && (ta[i] <= (tths[h]+dtth))){
            val.push_back(i);
            sband.push_back(band[i]);
          }
        }
      }
      
      // Get median for band
      median = Median(sband, sband.size());
      
      // Get Median Absolute Deviation
      for (i=0; i<sband.size(); i++){
        temp = abs(sband[i] - median);
        vmad.push_back(temp);
      }
      mad = Median(vmad, vmad.size()) * scale;

      // Masking
      for (i=0; i<val.size(); i++){
        //temp = abs(sband[i] - median) / mad;
        temp = (sband[i] - median) / mad;
        if (temp > eps){
          omask[val[i]] = 1.;
        }
      }

      val.clear();
      sband.clear();
      vmad.clear();
    }
}
