## MLP Architecture

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

## CNN Architecture

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

## RNN Architecture

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
