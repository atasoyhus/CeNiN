using System.IO;
using System.Runtime.InteropServices;

/*
 *--------------------------------------------------------------------------
 * CeNiN > MKLCBLAS.cs
 *--------------------------------------------------------------------------
 * CeNiN; a convolutional neural network implementation in pure C#
 * Huseyin Atasoy
 * huseyin @atasoyweb.net
 * http://huseyinatasoy.com
 * May 2019
 *--------------------------------------------------------------------------
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *--------------------------------------------------------------------------
 */

namespace CeNiN
{
    public unsafe class MKLCBLAS
    {
        [DllImport("mkl_rt.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void cblas_sgemm(
            int order, 
            int transA, 
            int transB,
            int m, 
            int n,
            int k,
            float alpha,
            float* A,
            int lda, 
            float* B, int ldb,
            float beta, 
            float* C,
            int ldc
        );

        [DllImport("mkl_rt.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void mkl_set_dynamic(int* d);

        [DllImport("mkl_rt.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void mkl_set_num_threads(int* n);

        [DllImport("mkl_rt.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int mkl_get_max_threads();

        public static void setNumThreads(int n)
        {
            int i = 0;
            mkl_set_dynamic(&i); // Disables dynamic adjustment of the number of threads...
            mkl_set_num_threads(&n);
        }

        public static void setDynamic()
        {
            int i = 1;
            mkl_set_dynamic(&i);
        }

        public static bool isAvailable()
        {
            string[] libs = { "mkl_rt.dll", "mkl_intel_thread.dll", "mkl_core.dll", "libiomp5md.dll", "mkl_def.dll", "mkl_avx.dll", "mkl_avx2.dll", "mkl_avx512.dll", "mkl_mc.dll", "mkl_mc3.dll" };
            for (int i = 0; i < libs.Length; i++)
                if (!File.Exists(libs[i]))
                    return false;

            return true;
        }

        public struct Order
        {
            public static int RowMajor = 101;
            public static int ColMajor = 102;
        }

        public struct Transpose
        {
            public static int None = 111;
            public static int Trans = 112;
            public static int ConjTrans = 113;
        }
    }
}
