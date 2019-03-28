using System;
using System.Runtime.InteropServices;

/*
 *--------------------------------------------------------------------------
 * CeNiN > Output.cs
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
    public class Output : Layer
    {
        public string[] classes;

        public string[] sortedClasses;
        public float[] probabilities;

        public Output(int[] inputTensorDims, string[] classes) : base(inputTensorDims)
        {
            type = "Output";
            this.classes = classes;
            probabilities = new float[inputTensorDims[2]];
            sortedClasses = new string[inputTensorDims[2]];
        }

        public unsafe string getDecision()
        {
            if (inputTensor.memPtr != null)
            {
                Array.Copy(classes, sortedClasses, classes.Length);
                Marshal.Copy((IntPtr)inputTensor.memPtr, probabilities, 0, classes.Length);
                Array.Sort(probabilities, sortedClasses);
                Array.Reverse(probabilities);
                Array.Reverse(sortedClasses);
                disposeInputTensor();
            }
            return sortedClasses[0];
        }

        public override void feedNext()
        {
            throw new NotImplementedException();
        }
    }
}
