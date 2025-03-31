<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Architectures</title>
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        table {
            width: 80%;
            border-collapse: collapse;
            margin: 20px auto;
        }
        th, td {
            border: 1px solid black;
            padding: 10px;
            text-align: center;
        }
        th {
            background-color: #d3d3d3;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        h2 {
            text-align: center;
        }
    </style>
</head>
<body>
    <h2>MLP Architecture</h2>
    <table>
        <tr>
            <th>Layer</th>
            <th>Size</th>
            <th>Filter</th>
            <th>Activation</th>
        </tr>
        <tr><td>Dense</td><td>128</td><td>--</td><td>relu</td></tr>
        <tr><td>Dense</td><td>64</td><td>--</td><td>relu</td></tr>
        <tr><td>Dense</td><td>N</td><td>--</td><td>softmax</td></tr>
    </table>

    <h2>CNN Architecture</h2>
    <table>
        <tr>
            <th>Layer</th>
            <th>Size</th>
            <th>Filter</th>
            <th>Activation</th>
        </tr>
        <tr><td>Convolution</td><td>128</td><td>3 × 3</td><td>relu</td></tr>
        <tr><td>Max Pooling</td><td>2</td><td>--</td><td>--</td></tr>
        <tr><td>Convolution</td><td>64</td><td>3 × 3</td><td>relu</td></tr>
        <tr><td>Max Pooling</td><td>2</td><td>--</td><td>--</td></tr>
        <tr><td>Convolution</td><td>32</td><td>3 × 3</td><td>relu</td></tr>
        <tr><td>Max Pooling</td><td>2</td><td>--</td><td>--</td></tr>
        <tr><td>Flatten</td><td>--</td><td>3 × 3</td><td>--</td></tr>
        <tr><td>Dense</td><td>128</td><td>--</td><td>relu</td></tr>
        <tr><td>Dense</td><td>64</td><td>--</td><td>relu</td></tr>
        <tr><td>Dense</td><td>N</td><td>--</td><td>softmax</td></tr>
    </table>

    <h2>RNN Architecture</h2>
    <table>
        <tr>
            <th>Layer</th>
            <th>Size</th>
            <th>Filter</th>
            <th>Activation</th>
        </tr>
        <tr><td>LSTM</td><td>128</td><td>--</td><td>--</td></tr>
        <tr><td>Dense</td><td>64</td><td>--</td><td>relu</td></tr>
        <tr><td>Dense</td><td>N</td><td>--</td><td>softmax</td></tr>
    </table>
</body>
</html>
