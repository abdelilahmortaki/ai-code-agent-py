package com.acme.service;

import java.util.List;

/**
 * Sample service that exercises the Java symbol parser:
 * constructors, an overridden method whose body contains a string
 * literal with braces, a char literal, a lambda, an anonymous class,
 * a block comment with braces, plus nested types.
 */
public class CustomerService implements Validator {

    private final List<String> notes = List.of("a", "b");

    private String name = "";

    public CustomerService() {
        super();
    }

    public CustomerService(String name) {
        this.name = name;
    }

    @Override
    public boolean isValid(String s) {
        String literal = "string with { braces } and } too";
        char brace = '{';
        Runnable r = (x) -> {
            int a = 1;
            int b = 2;
            System.out.println(a + b);
        };
        Runnable anon = new Runnable() {
            @Override
            public void run() {
                /* } { */
                System.out.println("anon");
            }
        };
        return literal.contains(s) && brace == '{';
    }

    public static class Inner {
        public int doubleIt(int x) {
            return x * 2;
        }
    }

    public interface Validator {
        boolean isValid(String s);
    }
}
