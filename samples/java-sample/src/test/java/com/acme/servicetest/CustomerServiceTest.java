package com.acme.servicetest;

import com.acme.service.CustomerService;

public class CustomerServiceTest {
    public void testFindCustomer() {
        CustomerService service = new CustomerService();
        String id = service.findCustomer("1");
    }
}
