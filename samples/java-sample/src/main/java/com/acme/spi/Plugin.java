package com.acme.spi;

public interface Plugin {
    String id();

    default String describe() {
        return "plugin " + id();
    }
}
