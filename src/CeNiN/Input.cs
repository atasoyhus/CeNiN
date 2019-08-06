using System.Drawing;
using System.Drawing.Imaging;

/*
 *--------------------------------------------------------------------------
 * CeNiN > Input.cs
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
    public unsafe class Input : Layer
    {
        public int[] inputSize;
        public float[] avgPixel;

        public enum ResizingMethod
        {
            Stretch,
            ZeroPad
        }

        private Bitmap resizedInputBmp;
        public Bitmap ResizedInputBmp
        {
            get
            {
                return resizedInputBmp;
            }
        }

        public Input(int[] inputTensorDims) : base(new int[] { 0, 0, 0 })
        {
            type = "Input";
            inputSize = (int[])inputTensorDims.Clone();
            avgPixel = new float[3];
        }

        public new void setOutputDims()
        {
            outputDims = (int[])inputSize.Clone();
        }

        private Bitmap resizeBitmap(Bitmap b, ResizingMethod resizingMethod)
        {
            Bitmap resizedBmp = new Bitmap(inputSize[1], inputSize[0], PixelFormat.Format24bppRgb);
            Graphics gr = Graphics.FromImage(resizedBmp);

            if (resizingMethod == ResizingMethod.Stretch)
                gr.DrawImage(b, 0, 0, inputSize[1], inputSize[0]);
            else
            {
                float inputAspRatio = inputSize[0] / inputSize[1];

                int newHeight, newWidth;
                float multiplier = (float)b.Width / b.Height;
                if (multiplier > inputAspRatio)
                {
                    multiplier = inputAspRatio / multiplier;
                    newWidth = inputSize[1];
                    newHeight = (int)(newWidth * multiplier);
                }
                else
                {
                    newHeight = inputSize[0];
                    newWidth = (int)(newHeight * multiplier);
                }

                gr.DrawImage(b, (inputSize[1] - newWidth) / 2.0f, (inputSize[0] - newHeight) / 2.0f, newWidth, newHeight);
            }

            gr.Dispose();

            return resizedBmp;
        }

        public void setInput(Bitmap input, ResizingMethod resizingMethod)
        {
            outputTensorMemAlloc();

            Bitmap iBitmap = (Bitmap)input.Clone();
            resizedInputBmp = resizeBitmap(iBitmap, resizingMethod);
            iBitmap.Dispose();
        }

        public override void feedNext()
        {
            BitmapData bmpData = resizedInputBmp.LockBits(new Rectangle(0, 0, inputSize[1], inputSize[0]), ImageLockMode.ReadOnly, resizedInputBmp.PixelFormat);
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

            resizedInputBmp.UnlockBits(bmpData);
            resizedInputBmp.Dispose();
        }
    }
}
