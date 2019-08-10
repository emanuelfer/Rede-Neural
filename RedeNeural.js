function sigmoid(x){
    return 1/(1+Math.exp(-x));
}

function dsigmoid(x){
    return x *(1 - x); 
}

class RedeNeural{
    constructor(i_nodes, h_nodes, o_nodes){
        this.i_nodes = i_nodes;
        this.h_nodes = h_nodes;
        this.o_nodes = o_nodes;

        this.bias_ih = new Matrix(h_nodes,1);
        this.bias_ih.randomize();
        this.bias_ho = new Matrix(o_nodes,1);
        this.bias_ho.randomize();

        this.weights_ih = new Matrix(this.h_nodes, this.i_nodes);
        this.weights_ih.randomize();

        this.weights_ho = new Matrix(this.o_nodes, this.h_nodes);
        this.weights_ho.randomize();

        this.learning_rate = 0.1;

    }

    train(arr, target){
        //input -> hidden
        let input = Matrix.arrayToMatrix(arr);
        let hidden = Matrix.multiply(this.weights_ih, input);
        hidden = Matrix.add(hidden, this.bias_ih);

        hidden.map(sigmoid);

        //hiden -> output

        let output = Matrix.multiply(this.weights_ho, hidden);
        output = Matrix.add(output, this.bias_ho);
        output.map(sigmoid);

        // backpropagtion
        //output -> hidden
        let expected = Matrix.arrayToMatrix(target);
        let output_error = Matrix.subtract(expected, output);
        let d_output = Matrix.map(output, dsigmoid);
        let hidden_T = Matrix.transpose(hidden);

        let gradient = Matrix.hadarmard(output_error, d_output);
        gradient = Matrix.escalar_multiply(gradient, this.learning_rate);

        this.bias_ho = Matrix.add(this.bias_ho, gradient);

        let weights_ho_deltas = Matrix.multiply(gradient, hidden_T);
        this.weights_ho = Matrix.add(this.weights_ho, weights_ho_deltas);

        //hidden -> input
        let weights_ho_T = Matrix.transpose(this.weights_ho);
        let hidden_error = Matrix.multiply(weights_ho_T, output_error);
        let d_hidden = Matrix.map(hidden,dsigmoid);
        let input_T = Matrix.transpose(input);

        let gradient_H = Matrix.hadarmard(hidden_error, d_hidden);
        gradient_H = Matrix.escalar_multiply(gradient_H, this.learning_rate);

        this.bias_ih = Matrix.add(this.bias_ih, gradient_H);

        let weights_ih_deltas = Matrix.multiply(gradient_H, input_T);
        this.weights_ih = Matrix.add(this.weights_ih, weights_ih_deltas);
    }

    predict(arr){
        //input -> hidden
        let input = Matrix.arrayToMatrix(arr);

        let hidden = Matrix.multiply(this.weights_ih, input);

        hidden = Matrix.add(hidden, this.bias_ih);
        hidden.map(sigmoid);

        //hiden -> output

        let output = Matrix.multiply(this.weights_ho, hidden);
        output = Matrix.add(output, this.bias_ho);
        output.map(sigmoid);
        output.print();
        output = Matrix.matrixToArray(output);

        return output;
    }
}