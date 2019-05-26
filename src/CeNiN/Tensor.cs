using System;
using System.Runtime.InteropServices;

/*
 *--------------------------------------------------------------------------
 * CeNiN > Tensor.cs
 *--------------------------------------------------------------------------
 * CeNiN; a convolutional neural network implementation in pure C#
 * Huseyin Atasoy
 * huseyin @atasoyweb.net
 * http://huseyinatasoy.com
 * March 2019
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
    unsafe public class Tensor
    {
        private int totalLength;
        private int[] dims;

        public float* memPtr;
        public int consumedMem;

        private int[] dimProds;

        public static bool useMKLCBLAS_GEMM = false;

        public int TotalLength
        {
            get
            {
                return totalLength;
            }
        }

        public int[] Dimensions
        {
            get
            {
                return dims;
            }
        }

        public Tensor(int[] dims)
        {
            totalLength = multAll(dims);

            this.dims = (int[])dims.Clone();
            updateDimProds();

            consumedMem = totalLength * sizeof(float);
            memPtr = (float*)Marshal.AllocHGlobal(consumedMem);
            for (int i = 0; i < totalLength; i++)
                memPtr[i] = 0.0f;
        }

        private void updateDimProds()
        {
            dimProds = new int[dims.Length];
            dimProds[0] = 1;
            for (int i = 1; i < dims.Length; i++)
                dimProds[i] = dimProds[i - 1] * dims[i - 1];
        }

        public float this[params int[] indexes]
        {
            get
            {
                int ind = get1DInd(indexes);
                return memPtr[ind];
            }
            set
            {
                int ind = get1DInd(indexes);
                memPtr[ind] = value;
            }
        }

        private int get1DInd(int[] indexes)
        {
            int ind = indexes[0];
            for (int i = 1; i < indexes.Length; i++)
                ind += dimProds[i] * indexes[i];

            if (ind >= totalLength)
                throw new IndexOutOfRangeException();
            return ind;
        }

        private static int multAll(int[] array)
        {
            int mul = 1;
            for (int i = 0; i < array.Length; i++)
                mul *= array[i];
            return mul;
        }

        public bool reshape(int[] newDims)
        {
            if (multAll(dims) != multAll(newDims))
                return false;
            dims = (int[])newDims.Clone();
            updateDimProds();
            return true;
        }

        private static bool dimsEqual(Tensor t1, Tensor t2)
        {
            if (t1.dims.Length != t2.dims.Length)
                return false;
            for (int i = 0; i < t1.dims.Length; i++)
                if (t1.dims[i] != t2.dims[i])
                    return false;
            return true;
        }

        public Tensor clone()
        {
            Tensor t = new Tensor(dims);
            for (int i = 0; i < totalLength; i++)
                t.memPtr[i] = memPtr[i];
            return t;
        }

        public static Tensor operator *(Tensor t1, float f)
        {
            Tensor t = new Tensor(t1.dims);
            for (int i = 0; i < t.totalLength; i++)
                t.memPtr[i] = t1.memPtr[i] * f;
            return t;
        }

        public static Tensor operator +(Tensor t1, float f)
        {
            Tensor t = new Tensor(t1.dims);
            for (int i = 0; i < t.totalLength; i++)
                t.memPtr[i] = t1.memPtr[i] + f;
            return t;
        }

        public static Tensor operator -(Tensor t1, float f)
        {
            return t1 + (-f);
        }

        public static Tensor operator +(Tensor t1, Tensor t2)
        {
            if (
                    t1.dims.Length == 2 && t2.dims.Length == 2 &&
                    (
                        (t1.dims[0] == 1 && t2.dims[1] == 1) ||
                        (t1.dims[1] == 1 && t2.dims[0] == 1)
                    )
                )
                return broadcastedAddition(t1, t2);

            if (!dimsEqual(t1, t2))
                return null;

            Tensor t = new Tensor(t1.dims);

            for (int i = 0; i < t1.totalLength; i++)
                t.memPtr[i] = t1.memPtr[i] + t2.memPtr[i];

            return t;
        }

        public void GEMM(Tensor A, Tensor B, float alpha, float beta)
        {
            int m = A.Dimensions[0];
            int n = B.Dimensions[1];
            int k = A.Dimensions[1];
            int lda = m, ldb = k, ldc = m;

            MKLCBLAS.cblas_sgemm(
                MKLCBLAS.Order.ColMajor,
                MKLCBLAS.Transpose.None,
                MKLCBLAS.Transpose.None,
                m, n, k,
                alpha, A.memPtr, lda,
                B.memPtr, ldb,
                beta, this.memPtr, ldc);
        }

        public static Tensor operator *(Tensor t1, Tensor t2)
        {
            if (t1.dims.Length != 2 || t2.dims.Length != 2 || t1.dims[1] != t2.dims[0])
                return null;

            Tensor t = new Tensor(new int[] { t1.dims[0], t2.dims[1] });

            if (useMKLCBLAS_GEMM)
            {
                t.GEMM(t1, t2, 1, 0);
                return t;
            }

            float sum;
            int[] ind1 = new int[] { 0, 0 };
            int[] ind2 = new int[] { 0, 0 };
            int[] ind3 = new int[] { 0, 0 };

            for (int i = 0; i < t1.dims[0]; i++)
            {
                ind1[0] = i;
                ind3[0] = i;
                for (int k = 0; k < t2.dims[1]; k++)
                {
                    ind2[1] = k;
                    ind3[1] = k;
                    sum = 0;
                    for (int j = 0; j < t1.dims[1]; j++)
                    {
                        ind1[1] = j;
                        ind2[0] = j;
                        sum += t1[ind1] * t2[ind2];
                    }
                    t[ind3] = sum;
                }
            }

            return t;
        }

        private static Tensor broadcastedAddition(Tensor t1, Tensor t2)
        {
            int dim1 = t1.totalLength;
            int dim2 = t2.totalLength;

            Tensor t = new Tensor(new int[] { dim1, dim2 });
            int[] ind = new int[] { 0, 0 };
            for (ind[0] = 0; ind[0] < dim1; ind[0]++)
                for (ind[1] = 0; ind[1] < dim2; ind[1]++)
                    t[ind] = t1.memPtr[ind[0]] + t2.memPtr[ind[1]];

            return t;
        }

        public void Dispose()
        {
            if (memPtr != null)
            {
                Marshal.FreeHGlobal((IntPtr)memPtr);
                memPtr = null;
            }
        }

        ~Tensor()
        {
            Dispose();
        }
    }
}
