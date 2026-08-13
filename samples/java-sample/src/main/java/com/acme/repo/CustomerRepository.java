package com.acme.repo;

import com.acme.model.Status;

public interface CustomerRepository {
    boolean exists(Status status);
}
