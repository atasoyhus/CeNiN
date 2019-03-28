using System;

/*
 *--------------------------------------------------------------------------
 * CeNiN > Conv_3.cs
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
    public class Conv_3 : Layer
    {
        public int[] stride;

        public Tensor weights;
        public Tensor biases;

        public Conv_3(int[] inputTensorDims, int[] pad) : base(inputTensorDims, pad)
        {
            type = "Convolution";
            stride = new int[2];
        }

        public new void setOutputDims()
        {
            int newHeight = (int)Math.Floor((double)(InputTensorDims[0] - weights.Dimensions[0]) / stride[0]) + 1;
            int newWidth = (int)Math.Floor((double)(InputTensorDims[1] - weights.Dimensions[1]) / stride[1]) + 1;

            outputDims = new int[] { newHeight, newWidth, weights.Dimensions[3] };
        }

        public unsafe override void feedNext()
        {
            outputTensorMemAlloc();

            int inputHeight = InputTensorDims[0];
            int inputWidth = InputTensorDims[1];
            int filterHeight = weights.Dimensions[0];
            int filterWidth = weights.Dimensions[1];
            int channelCount = weights.Dimensions[2];
            int filterCount = weights.Dimensions[3];

            int mCountH = inputHeight - filterHeight + 1;
            int mCountW = inputWidth - filterWidth + 1;
            Tensor possibleH = new Tensor(new int[] { outputDims[0], 1 });
            int j = 0;
            for (int i = 0; i < mCountH; i += stride[0])
                possibleH.memPtr[j++] = i;

            Tensor possibleW = new Tensor(new int[] { 1, outputDims[1] });
            j = 0;
            for (int i = 0; i < mCountW; i += stride[1])
                possibleW.memPtr[j++] = i;

            Tensor startingindexes = possibleW + possibleH * inputWidth;
            possibleH.Dispose();
            possibleW.Dispose();


            possibleH = new Tensor(new int[] { filterHeight, 1 });
            for (int i = 0; i < filterHeight; i++)
                possibleH.memPtr[i] = i;

            possibleW = new Tensor(new int[] { 1, filterWidth });
            for (int i = 0; i < filterWidth; i++)
                possibleW.memPtr[i] = i;

            Tensor offsets = possibleW + possibleH * inputWidth;
            possibleH.Dispose();
            possibleW.Dispose();

            startingindexes.reshape(new int[] { startingindexes.TotalLength, 1 });
            offsets.reshape(new int[] { 1, offsets.TotalLength });
            Tensor allindexes = startingindexes + offsets;
            startingindexes.Dispose();
            offsets.Dispose();

            int outputH_W = outputDims[0] * outputDims[1];

            Tensor allInOne = new Tensor(new int[] { outputH_W, filterHeight * filterWidth * channelCount });

            int h_W = inputHeight * inputWidth;
            int fH_fW = filterHeight * filterWidth;
            int h_w_fH_fW = h_W * fH_fW;
            int[] aiInd = new int[] { 0, 0 };
            int[] aioInd = new int[] { 0, 0 };
            int tmp;
            for (int ch = 0; ch < channelCount; ch++)
            {
                for (int k = 0; k < outputH_W; k++)
                {
                    aioInd[0] = k;
                    aiInd[0] = k;
                    for (int m = 0; m < fH_fW; m++)
                    {
                        aioInd[1] = ch * fH_fW + m;
                        aiInd[1] = m;
                        tmp = (int)allindexes[aiInd] + h_W * ch;
                        allInOne[aioInd] = inputTensor.memPtr[tmp];
                    }
                }
            }

            allindexes.Dispose();

            nextLayer.inputTensor.reshape(new int[] { allInOne.Dimensions[0], filterCount });

            float sum;
            int x = allInOne.Dimensions[1];
            int y = filterCount;
            int z = allInOne.Dimensions[0];
            int aioInd_;
            int outputInd_;
            int weightsInd_;
            for (int f = 0; f < y; f++)
            {
                for (int g = 0; g < z; g++)
                {
                    sum = 0;
                    for (int h = 0; h < x; h++)
                    {
                        aioInd_ = h * z + g;
                        weightsInd_ = f * x + h;
                        sum += weights.memPtr[weightsInd_] * allInOne.memPtr[aioInd_];
                    }
                    outputInd_ = f * z + g;
                    nextLayer.inputTensor.memPtr[outputInd_] = sum + biases.memPtr[f];
                }
            }

            nextLayer.inputTensor.reshape(outputDims);

            allInOne.Dispose();

            disposeInputTensor();
        }
    }
}
