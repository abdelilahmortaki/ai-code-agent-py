package com.acme.model;

public record Point(int x, int y) {
    public Point {
        validate();
    }

    private void validate() {
    }

    public int sum() {
        return x + y;
    }
}
