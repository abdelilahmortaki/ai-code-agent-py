package com.acme.model;

public enum Status {
    ACTIVE,
    PENDING(1, "waiting"),
    FAILED {
        @Override
        public String label() {
            return "failed!";
        }
    };

    private final int code;
    private final String labelText;

    Status() {
        this(0, "");
    }

    Status(int code, String labelText) {
        this.code = code;
        this.labelText = labelText;
    }

    public String label() {
        return labelText;
    }

    public int code() {
        return code;
    }
}
