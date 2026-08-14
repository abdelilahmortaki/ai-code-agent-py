package com.acme.model;

public class Contract {
    private final Status status;

    public Contract(Status status) {
        this.status = status;
    }

    public Status status() {
        return status;
    }
}
