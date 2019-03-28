using System;

/*
 *--------------------------------------------------------------------------
 * CeNiN > SoftMax.cs
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
    public unsafe class SoftMax : Layer
    {
        public SoftMax(int[] inputTensorDims) : base(inputTensorDims)
        {
            type = "SoftMax";
        }

        public override void feedNext()
        {
            outputTensorMemAlloc();

            float max = float.MinValue;
            for (int i = 0; i < inputTensor.TotalLength; i++)
                if (inputTensor.memPtr[i] > max)
                    max = inputTensor.memPtr[i];

            float sum = 0;
            float* nLMR = nextLayer.inputTensor.memPtr;
            for (int i = 0; i < inputTensor.TotalLength; i++)
            {
                nLMR[i] = (float)Math.Exp(inputTensor.memPtr[i] - max);
                sum += nLMR[i];
            }

            for (int i = 0; i < inputTensor.TotalLength; i++)
                nLMR[i] /= sum;

            disposeInputTensor();
        }
    }
}
