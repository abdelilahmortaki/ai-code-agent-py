package com.acme.api;

import com.acme.service.CustomerService;

public class CustomerController {
    private final CustomerService service;

    public CustomerController(CustomerService service) {
        this.service = service;
    }

    public String show(String id) {
        return service.findCustomer(id);
    }
}
