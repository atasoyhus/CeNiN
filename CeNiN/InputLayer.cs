using System.Drawing;
using System.Drawing.Imaging;

/*
 *--------------------------------------------------------------------------
 * CeNiN > InputLayer.cs
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
    public unsafe class InputLayer : Layer
    {
        public int layerCount;
        public int[] inputSize;
        public float[] avgPixel;

        private Bitmap inputImage;

        public InputLayer(int[] inputTensorDims) : base(new int[] { 0, 0, 0 })
        {
            type = "Input";
            inputSize = (int[])inputTensorDims.Clone();
            avgPixel = new float[3];
        }

        public new void setOutputDims()
        {
            outputDims = (int[])inputSize.Clone();
        }

        public void setInput(Bitmap bmp)
        {
            inputImage = (Bitmap)bmp.Clone();
        }

        public override void feedNext()
        {
            outputTensorMemAlloc();
            Bitmap resizedBmp = new Bitmap(inputSize[1], inputSize[0], PixelFormat.Format24bppRgb);
            Graphics gr = Graphics.FromImage(resizedBmp);
            gr.DrawImage(inputImage, 0, 0, inputSize[1], inputSize[0]);
            gr.Dispose();
            inputImage.Dispose();

            BitmapData bmpData = resizedBmp.LockBits(new Rectangle(0, 0, inputSize[1], inputSize[0]), ImageLockMode.ReadOnly, resizedBmp.PixelFormat);
            int stride = bmpData.Stride;
            int emptyBytesCount = stride - bmpData.Width * 3;
            int rowLengthWithoutEB = stride - emptyBytesCount;
            byte* dataPtr = (byte*)bmpData.Scan0.ToPointer();
            int byteCount = stride * bmpData.Height;

            int i = 0;
            int pControl = 0;
            int b, g, r;
            int[] ind = new int[] { 0, 0, 0 };
            while (i < byteCount)
            {
                b = *dataPtr;
                dataPtr++;
                g = *dataPtr;
                dataPtr++;
                r = *dataPtr;
                dataPtr++;

                ind[2] = 0;
                writeNextLayerInput(ind, r - avgPixel[0]);
                ind[2] = 1;
                writeNextLayerInput(ind, g - avgPixel[1]);
                ind[2] = 2;
                writeNextLayerInput(ind, b - avgPixel[2]);

                ind[1] += 1;

                pControl += 3;
                i += 3;
                if (pControl == rowLengthWithoutEB)
                {
                    pControl = 0;
                    dataPtr += emptyBytesCount;

                    ind[1] = 0;
                    ind[0] += 1;
                }
            }

            resizedBmp.UnlockBits(bmpData);
            resizedBmp.Dispose();
        }
    }
}
