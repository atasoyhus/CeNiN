
Imports CeNiN

'--------------------------------------------------------------------------
' CeNiN; a convolutional neural network implementation in pure C#
' Huseyin Atasoy
' huseyin @atasoyweb.net
' http://huseyinatasoy.com
' March 2019
'--------------------------------------------------------------------------
' Copyright 2019 Huseyin Atasoy
'
' Licensed under the Apache License, Version 2.0 (the "License");
' you may Not use this file except in compliance with the License.
' You may obtain a copy of the License at
'
' http://www.apache.org/licenses/LICENSE-2.0
'
' Unless required by applicable law Or agreed to in writing, software
' distributed under the License Is distributed on an "AS IS" BASIS,
' WITHOUT WARRANTIES Or CONDITIONS OF ANY KIND, either express Or implied.
' See the License for the specific language governing permissions And
' limitations under the License.
'--------------------------------------------------------------------------

Public Class Form1
    Private DateTime As DateTime
    Private cnn As CNN

    Private Sub Button1_Click(ByVal sender As Object, ByVal e As EventArgs) Handles Button1.Click
        Dim OPF As OpenFileDialog = New OpenFileDialog()
        OPF.Filter = "CeNiN file|*.cenin"
        If OPF.ShowDialog() <> DialogResult.OK Then Return
        TextBox1.Clear()
        PrependLine("Parsing CeNiN file...")
        Application.DoEvents()
        Tic()
        cnn = New CNN(OPF.FileName)
        PrependLine(cnn.layerCount & "+2 layers, " _
                & cnn.totalWeightCount & " weights and" _
                & cnn.totalBiasCount & " biases were loaded in " _
                & Toc() & " seconds.")
        Button2.Enabled = True
    End Sub

    Private Sub Button2_Click(ByVal sender As Object, ByVal e As EventArgs) Handles Button2.Click
        Dim opf As OpenFileDialog = New OpenFileDialog()
        opf.Filter = "Image files|*.bmp;*.jpeg;*.jpg;*.png"
        If opf.ShowDialog() <> DialogResult.OK Then Return
        Dim b As Bitmap = New Bitmap(opf.FileName)
        cnn.inputLayer.setInput(b, Input.ResizingMethod.ZeroPad)
        PictureBox1.Image = cnn.inputLayer.ResizedInputBmp.Clone()
        Tic()
        Dim CurrentLayer As Layer = cnn.inputLayer
        Dim i As Integer = 0

        While CurrentLayer.nextLayer IsNot Nothing
            If i = 0 Then
                PrependLine("Loading bitmap data...")
            Else
                PrependLine("Layer " & i & " (" & CurrentLayer.type & ") ...")
            End If

            Application.DoEvents()
            CurrentLayer.feedNext()
            CurrentLayer = CurrentLayer.nextLayer
            i += 1
        End While

        Dim OutputLayer As Output = CType(CurrentLayer, Output)
        PrependLine("Finished in " & Toc().ToString() & " seconds")

        Dim Decision As String = OutputLayer.getDecision()
        Dim HLine As String = New String("-"c, 100)
        PrependLine(HLine, "")

        For i = 2 To 0 Step -1
            PrependLine(" #" & (i + 1) & "   " & OutputLayer.sortedClasses(i) & " (" & Math.Round(OutputLayer.probabilities(i), 3) & ")", "")
        Next

        PrependLine(HLine, "")
        PrependLine("THE HIGHEST 3 PROBABILITIES: ", "")
        PrependLine(HLine, "")
        PrependLine("DECISION: " & Decision)
        PrependLine(HLine, "")
    End Sub

    Private Function Tic() As DateTime
        DateTime = DateTime.Now
        Return DateTime
    End Function

    Private Function Toc() As Double
        Return Math.Round((DateTime.Now - DateTime).TotalSeconds, 3)
    End Function

    Private Sub PrependLine(ByVal Text As String, ByVal Optional Prefix As String = "-->  ")
        TextBox1.Text = Prefix & Text & vbCrLf & TextBox1.Text
    End Sub
End Class
